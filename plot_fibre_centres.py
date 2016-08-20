#!/usr/bin/env python3
# encoding: utf-8
#
# plot_fibre_centres.py
#
# Created by José Sánchez-Gallego on Aug 14, 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse
import os
from pathlib import Path
import sys
import warnings

import astropy.table as table
from astropy.units import UnitsWarning

import numpy as np

import progressbar

import matplotlib.pyplot as plt

from guider_utils import GuiderImage


warnings.filterwarnings('ignore', category=UnitsWarning)


def compare_with_flat(gimg, save=True, plot=True):
    """Compares the current result for a flat with the processed image."""

    flat_fibers = gimg.flat.fibers

    fiber_offset = []
    fibers_used = []
    for gprobe in flat_fibers:

        if gprobe['gprobe_id'] in [3, 11]:
            continue

        if np.isnan(gprobe['xcen']) or np.isnan(gprobe['ycen']):
            continue

        fiber = gimg.fibers[gimg.fibers['gprobe_id'] == gprobe['gprobe_id']]

        if len(fiber) == 0:
            continue

        offset = np.sqrt((fiber['xcen'][0] - gprobe['xcen']) ** 2 +
                         (fiber['ycen'][0] - gprobe['ycen']) ** 2)

        fiber_offset.append(offset)
        fibers_used.append(fiber['gprobe_id'][0])

    if len(fiber_offset) < 5:
        return False

    if plot:
        fig, ax = plt.subplots()
        ax.imshow(gimg.mask, origin='lower')
        for gprobe_id in fibers_used:
            fiber = gimg.fibers[gimg.fibers['gprobe_id'] == gprobe_id]
            plt.scatter(fiber['xcen'][0], fiber['ycen'][0], marker='x',
                        color='k', s=10)
        plt.savefig(gimg.path.name.split('.')[0] + '.png')

    mean_offset = np.mean(fiber_offset)

    if save and mean_offset:
        unit = open('offsets.dat', 'a')
        unit.write('{0}   {1}   {2:.5f}\n'.format(gimg.path.name,
                                                  gimg.flat.path.name,
                                                  mean_offset))
        unit.close()

    return mean_offset


def plot_fibre_centres(gimg_path):
    """Plots the fibre centres from a full night to check for flextures."""

    gimg_path = Path(gimg_path)
    assert gimg_path.exists()

    gimgs = list(gimg_path.glob('gimg-*.fits.gz'))

    bar = progressbar.ProgressBar(redirect_stdout=True, max_value=len(gimgs))

    for ii, gimg in enumerate(gimgs):
        gimg_obj = GuiderImage(gimg, do_process=True, verbose=False)

        if gimg_obj.invalid:
            continue

        if gimg_obj.type == 'object':
            value = compare_with_flat(gimg_obj)
            print('object', gimg_obj.path.name, value if value else None)

        # bar.update(ii)


def plot_offsets():

    offsets = table.Table.read('offsets.dat', format='ascii',
                               names=['image', 'flat', 'offset'])

    fig, ax = plt.subplots()

    xx = [int(image.split('.')[0].split('-')[1]) for image in offsets['image']]
    ax.scatter(xx, offsets['offset'], marker='.', color='k', s=2)
    ax.plot(xx, offsets['offset'], 'k--', lw=0.6)

    flats = np.unique([int(image.split('.')[0].split('-')[1])
                       for image in offsets['flat']])

    ax.vlines(flats, 0.0, 3.0, colors='r')

    plt.savefig('offsets.pdf')


def main():

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))

    parser.add_argument('PATH', metavar='PATH', type=str,
                        help='The path with the images to analyse.s')

    args = parser.parse_args()

    # plot_fibre_centres(args.PATH)
    plot_offsets()


if __name__ == '__main__':
    main()
