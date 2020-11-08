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


ap = argparse.ArgumentParser(description='Plot fits file. Output is basename + pol + .png')

ap.add_argument("filename", type=str, help="FITS filename to plot")
ap.add_argument("--outdir", type=str, default='./', help='Output path to write file')
ap.add_argument("--subfile", type=str, default=None, help="file to subtract from filename before plotting")

ap.add_argument("--pol_index", type=int, default=0, help="polarization index in fits file to plot")
ap.add_argument("--cmap", type=str, default='bone_r', help='colormap')
ap.add_argument("--vmin", type=float, default=0, help="cmap vmin")
ap.add_argument("--vmax", type=float, default=15, help="cmap vmax")
ap.add_argument("--radius", type=float, default=20, help="radius from center of plot edge")

if __name__ == "__main__":

    # parse args
    a = ap.parse_args()

    # open fits and get wcs
    hdu = fits.open(a.filename)[0]
    head = hdu.header
    data = hdu.data
    wcs = WCS(hdu, naxis=2)

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

    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111, projection=wcs)
    xax, yax = ax.coords[0], ax.coords[1]

    cax = ax.imshow(data[a.pol_index, 0], aspect='auto', origin='lower', vmin=a.vmin, vmax=a.vmax, cmap=a.cmap)
    ax.set_xlabel(r'Right Ascension', fontsize=16, labelpad=0.75)
    ax.set_ylabel(r'Declination', fontsize=16, labelpad=0.75)
    casa_utils.set_xlim(ax, wcs, xlim, center[1])
    casa_utils.set_ylim(ax, wcs, ylim, center[0])
    ax.tick_params(labelsize=16, direction='in')
    xax.tick_params(length=5); yax.tick_params(length=5)
    xax.set_ticks_position('b'); yax.set_ticks_position('l')
    xax.set_major_formatter('d')
    bmaj, bmin, bpa = casa_utils.get_beam_info(h1)
    casa_utils.plot_beam(ax, wcs, bmaj, bmin, bpa, frac=np.max([.15 - (a.radius-5)/350, .02]), pad=1.5)
    cbax, cbar = casa_utils.top_cbar(fig, ax, cax, size='5%', label='Jy/beam', pad=0.1, length=5, labelsize=16, fontsize=20, minpad=1)
    ax.grid(color='k', ls='--')
    ax.text(0.03, 0.88, "{} polarization\n{:.1f} MHz".format(pols[a.pol_index], freq/1e6),
            fontsize=15, color='k', transform=ax.transAxes,
            bbox=dict(boxstyle='square', fc='w', ec='None', alpha=0.8, pad=.2))

    fname = os.path.splitext(os.path.basename(a.filename))
    fname = os.path.join(a.outdir, fname[0] + '.{}.png'.format(pols[a.pol_index]))
    fig.savefig(fname, dpi=100, bbox_inches='tight')
