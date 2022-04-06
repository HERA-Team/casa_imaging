#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from pyuvdata import UVData, utils as uvutils
import numpy as np
import argparse
from casa_imaging import casa_utils
import os


ap = argparse.ArgumentParser(description='Plot fits file. Output is basename.png.')

ap.add_argument("filename", type=str, help="FITS filename to plot")
ap.add_argument("--outdir", type=str, default='./', help='Output path to write file')
ap.add_argument("--subfile", type=str, default=None, help="file to subtract from filename before plotting")

ap.add_argument("--cmap", type=str, default='bone_r', help='colormap. Can feed as "cmap1,cmap2,.." for each pol in file.')
ap.add_argument("--vmin", type=str, default='0', help='cmap vmin. Can feed as "vmin1,vmin2,.." for each pol in file.')
ap.add_argument("--vmax", type=str, default='15', help='cmap vmax. Can feed as "vmax1,vmax2,.." for each pol in file.')
ap.add_argument("--radius", type=float, default=20, help="radius from center of plot edge")

if __name__ == "__main__":

    # parse args
    a = ap.parse_args()

    # open fits and get wcs
    hdu = fits.open(a.filename)
    head = hdu[0].header
    data = hdu[0].data
    wcs = WCS(hdu[0], naxis=2)

    # subtract subfile
    if a.subfile is not None:
        data -= fits.open(a.filename)[0].data

    # get image properties
    Npix = head['NAXIS1']
    center = wcs.wcs_pix2world([[Npix//2, Npix//2]], 1).squeeze()
    xlim = (center[0] + a.radius, center[0] - a.radius)
    ylim = (center[1] - a.radius, center[1] + a.radius)
    freq = head['CRVAL3']
    pols = int(head['CRVAL4']) + np.arange(int(head['NAXIS4'])) * int(head["CDELT4"])
    pols = [uvutils.polnum2str(p) for p in pols]
    Npols = len(pols)

    # parse cmap arguments
    cmap, vmin, vmax = a.cmap.split(','), a.vmin.split(','), a.vmax.split(',')
    if len(cmap) == 1:
        cmap = cmap * Npols
    if len(vmin) == 1:
        vmin = vmin * Npols
    if len(vmax) == 1:
        vmax = vmax * Npols
    vmin, vmax = [float(vm) for vm in vmin], [float(vm) for vm in vmax]

    fig = plt.figure(figsize=(5 * Npols, 5), dpi=100)
    fig.subplots_adjust(wspace=0.1)
    # iterate over pol
    for i, pol in enumerate(pols):
        ax = fig.add_subplot(int("1{}{}".format(Npols, i+1)), projection=wcs)
        xax, yax = ax.coords[0], ax.coords[1]
        cax = ax.imshow(data[i, 0], aspect='auto', origin='lower', vmin=vmin[i], vmax=vmax[i], cmap=cmap[i])
        casa_utils.set_xlim(ax, wcs, xlim, center[1])
        casa_utils.set_ylim(ax, wcs, ylim, center[0])
        ax.tick_params(labelsize=14, direction='in')
        xax.tick_params(length=5); yax.tick_params(length=5)
        xax.set_ticks_position('b'); yax.set_ticks_position('l')
        xax.set_major_formatter('d')
        ax.set_xlabel(r'Right Ascension', fontsize=16, labelpad=0.75)
        if i == 0:
            ax.set_ylabel(r'Declination', fontsize=16, labelpad=0.75)
        else:
            yax.set_ticklabel_visible(False)
        bmaj, bmin, bpa = casa_utils.get_beam_info(hdu)
        if i == 0:
            casa_utils.plot_beam(ax, wcs, bmaj, bmin, bpa, frac=np.max([.15 - (a.radius-5)/350, .02]), pad=1.5)
        cbax, cbar = casa_utils.top_cbar(fig, ax, cax, size='5%', label='Jy/beam', pad=0.1, length=5, labelsize=14, fontsize=16, minpad=1)
        ax.grid(color='k', ls='--')
        tag = "{} polarization\n{:.1f} MHz".format(pols[i], freq/1e6)
        ax.text(0.03, 0.88, tag, fontsize=15, color='k', transform=ax.transAxes,
                bbox=dict(boxstyle='square', fc='w', ec='None', alpha=0.8, pad=.2))

    fname = os.path.splitext(os.path.basename(a.filename))
    fname = os.path.join(a.outdir, fname[0] + '.png')
    fig.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
