#!/usr/bin/env python3
# encoding: utf-8
#
# guider_utils.py
#
# Created by José Sánchez-Gallego on Aug 14, 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from pathlib import Path
import warnings

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker

import astropy.io.fits as fits
import astropy.table as table

import numpy as np
from scipy.ndimage import measurements
from scipy.ndimage import morphology

from skimage import measure as skmeasure


def _warning(message, category=UserWarning, filename='', lineno=-1):
    print('{0}: {1}'.format(category.__name__, message))

warnings.showwarning = _warning

warnings.filterwarnings(
    'ignore',  message='Skipped unsupported reflection of expression-based')


def bin_image(img, bin_factor):
    """Return an image rebinned by BINxBIN pixels."""

    binned = np.zeros((img.shape[0] / bin_factor, img.shape[1] / bin_factor),
                      np.float32)

    for ii in np.arange(bin_factor):
        for jj in np.arange(bin_factor):
            binned += img[ii::bin_factor, jj::bin_factor]

    binned /= (bin_factor * bin_factor)

    return binned


class GuiderConfig(object):

    def __init__(self):

        self.bias = None
        self.dark = None
        self.flat = None

        self.get_gprobes()

    def get_gprobes(self):
        """Creates an astropy table of gprobes for each cart."""

        connection_parameters = {'username': 'sdssdb_admin',
                                 'password': '',
                                 'host': 'localhost',
                                 'port': 5432,
                                 'db_name': 'lcodb'}

        engine = create_engine(
            'postgresql://{username:s}:{password:s}@{host:s}:{port:d}/{db_name:s}'
            .format(**connection_parameters))

        metadata = MetaData(schema='platedb')
        metadata.reflect(engine)
        Base = automap_base(metadata=metadata)
        Base.prepare()

        GProbe = Base.classes.gprobe
        GProbe_columns = GProbe.__table__.columns.keys()

        # Creates a session
        Session = sessionmaker(engine, autocommit=True)
        session = Session()

        gprobes = session.query(GProbe).filter(
            GProbe.fiber_type != 'TRITIUM').order_by(GProbe.gprobe_id).all()
        gprobes_data = [[gprobe.cartridge.number] +
                        [getattr(gprobe, column) for column in GProbe_columns]
                        for gprobe in gprobes]
        self.gprobes = table.Table(rows=gprobes_data,
                                   names=['cartridge'] + GProbe_columns)


guiderConfig = GuiderConfig()


class GuiderImage(object):

    def __init__(self, image_path, do_process=True, verbose=True):
        """A class representing a guider image."""

        self.path = Path(image_path)
        self.hdulist = fits.open(self.path)

        self.fibers = None
        self.invalid = False
        self.verbose = verbose

        self.type = None
        self.binning = None
        self.expTime = None
        self._classify_image()

        self.dark_path = None
        self.processed_image = None
        self.processed_expTime = None

        if do_process:
            self.process()

    def _classify_image(self):
        """Detects the type of image and sets attributes accordingly."""

        assert 'proc' not in self.path.name, 'This is a processed image.'
        assert len(self.hdulist) == 1, 'More than one extension found.'

        self.type = self.hdulist[0].header['IMAGETYP'].strip().lower()

        self.binning = 1 if self.type == 'flat' else 2
        self.expTime = self.hdulist[0].header['EXPTIME']

    def process(self):
        """Processes an image."""

        self._do_bias()

        if self.type in ['dark', 'object']:
            self._do_dark()

        # Checks that we have the right flat for an object
        if self.type == 'object':
            flatfile = self.hdulist[0].header['FLATFILE']
            gimg_flat_path = self.path.parent / Path(flatfile).name

            if gimg_flat_path.absolute() != guiderConfig.flat.path.absolute():
                if self.verbose:
                    warnings.warn('replacing flat image from {0} to {1}'.format(
                        guiderConfig.flat.path, gimg_flat_path))
                    GuiderImage(gimg_flat_path, do_process=True,
                                verbose=self.verbose)

            self.flat = guiderConfig.flat

        if self.type in ['flat', 'object']:
            self.find_fibres()

    def _do_bias(self):

        if self.type == 'bias':
            self.bias_path = self.path.absolute()
            self.bias_level = np.median(self.hdulist[0].data)
            guiderConfig.bias = self

        else:

            try:
                biasfile = self.hdulist[0].header['BIASFILE']
                gimg_bias_path = self.path.parent / Path(biasfile).name
            except Exception:
                gimg_bias_path = None
                if self.verbose:
                    warnings.warn('image {0} does not have BIASFILE'.format(self.path))

            if gimg_bias_path and gimg_bias_path.absolute() != guiderConfig.bias.path.absolute():
                if self.verbose:
                    warnings.warn('replacing bias image from {0} to {1}'.format(
                        guiderConfig.bias.path, gimg_bias_path))
                GuiderImage(gimg_bias_path, do_process=True)
                self._do_bias()
                return
            else:
                self.processed_image = self.hdulist[0].data - guiderConfig.bias.bias_level

    def _do_dark(self):

        if self.type == 'dark':
            self.dark_path = self.path.absolute()
            guiderConfig.dark_path = self.dark_path
            self.processed_image = self.hdulist[0].data / self.expTime
            self.processed_expTime = 1.
            guiderConfig.dark = self

        else:

            try:
                darkfile = self.hdulist[0].header['DARKFILE']
                gimg_dark_path = self.path.parent / Path(darkfile).name
            except Exception:
                gimg_dark_path = None
                if self.verbose:
                    warnings.warn('image {0} does not have DARKFILE'.format(self.path))

            if gimg_dark_path and gimg_dark_path.absolute() != guiderConfig.dark.path.absolute():
                if self.verbose:
                    warnings.warn('replacing dark image from {0} to {1}'.format(
                        guiderConfig.dark.path, gimg_dark_path))
                GuiderImage(gimg_dark_path, do_process=True)
                self._do_dark()
                return
            else:
                self.processed_image = (self.processed_image -
                                        guiderConfig.dark.processed_image * self.expTime)

    def find_fibres(self):
        """Finds fibres in an image and matches them to the gprobes."""

        self.fibers = None

        image = self.processed_image

        median = np.median(image)

        if self.type == 'object':
            pk = np.percentile(image, 99.0, interpolation='higher')
        else:
            pk = np.percentile(image, 99.8, interpolation='linear')

        thresh = (median + pk) / 2.
        threshold_image = (image > thresh)

        # Fix nicks in fibers, and fill holes (necessary for the acquisition fibers)
        threshold_image = morphology.binary_closing(threshold_image, iterations=10)

        background = np.median(image[np.logical_not(threshold_image)])

        self.mask = morphology.binary_erosion(threshold_image, iterations=3)

        # Forces the threshold_image to be at least twice the level of the background
        if np.median(image[threshold_image]) < 2 * background:
            if self.verbose:
                warnings.warn('threshold level is less than twice the '
                              'background for file {0}. Marking it invalid.'
                              .format(self.path.name), UserWarning)
            self.invalid = True
            return

        (fiber_labels, nlabels) = measurements.label(threshold_image)
        raw_regions = skmeasure.regionprops(fiber_labels)

        if self.type == 'object':
            self.mask = np.zeros(image.shape)
            valid_regions = [reg for reg in raw_regions
                             if reg.area > 100 and reg.eccentricity < 0.2]
            for region in valid_regions:
                self.mask += fiber_labels == region.label
        else:
            valid_regions = raw_regions

        fibers = table.Table(None, names=['fiber_id', 'gprobe_id',
                                          'xcen', 'ycen', 'radius', 'npix'],
                             dtype=[int, int, float, float, float, int])

        binFactor = 2 if self.binning == 1 else 1  # Yep, this is stupid.
        for ii in range(len(valid_regions)):

            # obji = (fiber_labels == ii)
            obji = valid_regions[ii]

            npix = obji.area
            (yc, xc) = obji.centroid
            radius = np.sqrt(npix / np.pi)

            if binFactor == 2:
                xc = xc / binFactor - 0.5
                yc = yc / binFactor - 0.5

            radius /= binFactor

            fibers.add_row((ii + 1, -1, xc, yc, radius, npix))

        # Does the matching between gprobes and fibres.
        for gprobe in guiderConfig.gprobes:

            unassigned_fibers = fibers[fibers['gprobe_id'] == -1]
            if len(unassigned_fibers) == 0:
                break

            distances = np.sqrt(
                ((unassigned_fibers['xcen'] - float(gprobe['x_center'])) ** 2 +
                 (unassigned_fibers['ycen'] - float(gprobe['y_center'])) ** 2))

            matched_fiber = unassigned_fibers[np.argmin(distances)]
            matched_fiber_distance = np.min(distances)
            if matched_fiber_distance > float(gprobe['radius']):
                continue

            # Uses fiber_id to find the right fiber to assign it the gprobe_id
            matched_fiber_idx = fibers['fiber_id'] == matched_fiber['fiber_id']
            fibers['gprobe_id'][matched_fiber_idx] = gprobe['gprobe_id']

        # For a flat, let's require at least 10 fibres
        if self.type == 'flat' and len(fibers) < 10:
            if self.verbose:
                warnings.warn('flat {0} has fewer than 10 fibres'.format(self.path.name),
                              UserWarning)
            self.invalid = True
            return

        if len(fibers[fibers['gprobe_id'] == -1]) > 0:
            if self.verbose:
                warnings.warn('unasigned fibres for file {0}. Marking it as invalid.'
                              .format(self.path.name), UserWarning)
            self.invalid = True
            return

        # fiber_id is only useful for the matching with unassigned_fibers.
        # Now we can remove that column.
        fibers.remove_column('fiber_id')
        fibers.sort('gprobe_id')

        self.fibers = fibers

        if self.type == 'flat':
            guiderConfig.flat = self
