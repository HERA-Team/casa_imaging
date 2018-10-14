#!/usr/bin/env python2.7
"""
find_soures.py
==================

Given a CASA MFS image and model file, smooth the model
with the restoring beam of the image and identify peaks
to find sources and write them to file.
"""
import astropy.io.fits as fits
import argparse
import os
import sys
import shutil
import glob
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from scipy import signal
from casa_imaging import casa_utils as utils

args = argparse.ArgumentParser(description="Identify sources in an MFS model and image file.")

# IO Arguments
args.add_argument("modfile", type=str, help="CASA model FITS file. If multiple polarization or frequencies are present, they are first averaged over.")
args.add_argument("--imfile", default=None, type=str, help="CASA image FITS file to get beam info from. Default is a single pixel.")
args.add_argument("--outfile", default=None, type=str, help="Output file to write source info to in tab-delimited format.")
args.add_argument("--overwrite", default=False, action='store_true', help="overwrite output file.")
# Analysis Arguments
args.add_argument("--rb_Npx", default=31, type=int, help="Size of restoring beam model side-length in pixels. Should be odd valued.")
args.add_argument("--thresh", default=1.0, type=float, help="Flux threshold to stop iteration.")
args.add_argument("--maxiter", default=50, type=int, help="Maximum number of iterations (sources) to identify.")
args.add_argument("--trim_sources", default=True, type=bool, help="Exclude weaker sources within a synthesized beam of a stronger source.")
args.add_argument("--trim_scale", default=0.5, type=float, help="Multiplier of synthesized beam FWHM level, within which to trim weaker sources.")
args.add_argument("--plot", default=False, action='store_true', help="Make diagnostic plot")

if __name__ == "__main__":

    # parse args
    a = args.parse_args()

    # check output
    if a.outfile is None:
        outfile = os.path.splitext(a.modfile)
        outfile = "{}.srcs.tab".format(outfile[0])
        if os.path.exists(outfile) and not a.overwrite:
            raise ValueError("{} already exists and overwrite is False...".format(outfile))
    else:
        outfile = a.outfile
    plot_outfile = os.path.splitext(outfile)[0] + ".png"
    casa_reg_outfile = os.path.splitext(outfile)[0] + ".crtf"

    # load model file
    mhdu = fits.open(a.modfile)
    mhead = mhdu[0].header
    mwcs = WCS(mhead, naxis=2)

    # get hdu info
    pols, freqs, stokax, freqax = utils.get_hdu_info(mhdu)

    # load image file for beam info, use zeroth polarization if multi-pol available
    if a.imfile is not None:
        ihdu = fits.open(a.imfile)
        # get bmaj and bmin in pixel units
        bmaj, bmin, bpa = utils.get_beam_info(ihdu, pol_ind=0, pxunits=True)
    else:
        # else bmaj and bmin are single pixels
        bmaj = 1
        bmin = 1
        bpa = 0

    # get restoring beam
    rest_beam = utils.make_restoring_beam(bmaj, bmin, bpa, size=a.rb_Npx)

    # smooth model with restoring beam
    model = signal.convolve2d(np.mean(mhdu[0].data, axis=(0, 1)), rest_beam, mode='same')
    _model = model.copy()

    # make a source mask
    mask = np.zeros_like(model, dtype=np.bool)

    # iterate to get sources
    peak = np.nanmax(model)
    source_pixels = []
    source_peaks = []
    i = 0
    while peak > a.thresh:
        if i >= a.maxiter:
            break

        # get pixel at peak flux        
        pxl = np.array(np.where(model == peak))[:, 0]

        # ensure this pixel is not masked already
        trim = False
        if a.trim_sources:
            if mask[pxl[0], pxl[1]]:
                trim = True

        # subtract from model
        utils.subtract_beam(model, rest_beam*peak*(1-1e-5), pxl, subtract=True, inplace=True)

        if not trim:
            # add to mask
            utils.subtract_beam(mask, rest_beam > 0.5*a.trim_scale, pxl, subtract=False, inplace=True)

            # append
            source_peaks.append(peak)
            source_pixels.append(pxl)

        # get new peak
        peak = np.nanmax(model)
        i += 1

    source_peaks = np.array(source_peaks)
    source_pixels = np.array(source_pixels)

    # get ra and dec of pixels
    source_coords = mwcs.all_pix2world(source_pixels[:, ::-1], 0)

    # write to file
    np.savetxt(outfile, np.concatenate([source_pixels, source_coords, source_peaks[:, None]], axis=1),
               fmt="%08.4f", delimiter='\t', header='RA [px]\tDec [px]\tRA [deg]\tDec [deg]\tPeak_flux')
    with open(casa_reg_outfile, 'w') as f:
        f.write('#CRTFv0 CASA Region Text Format version 0\n')
        reg = "ellipse[[{:0.3f}deg, {:0.3f}deg], [{:0.3f}deg, {:0.3f}deg], {:0.3f}deg] coord=J2000\n"
        # convert bmaj and bmin back to degree
        bmaj *= np.abs(mhdu[0].header["CDELT1"])
        bmin *= np.abs(mhdu[0].header["CDELT1"])
        for i in range(len(source_peaks)):
            f.write(reg.format(source_coords[i, 0], source_coords[i, 1], bmaj, bmin, bpa))

    # plot
    if a.plot:
        fig = plt.figure(figsize=(14, 6))
        fig.subplots_adjust(hspace=0.2)

        ax = fig.add_subplot(121, projection=mwcs)
        cax = ax.imshow(_model, origin='lower', cmap='viridis')
        fig.colorbar(cax, ax=ax)
        ax.plot(source_pixels[:, 1], source_pixels[:, 0], marker='*', markersize=4, color='w', ls='')
        ax.set_title("Smoothed Model", fontsize=10)
        ax.set_xlabel("Right Ascension", fontsize=10)
        ax.set_ylabel("Declination", fontsize=10)

        ax = fig.add_subplot(122, projection=mwcs)
        cax = ax.imshow(model, origin='lower', cmap='viridis')
        fig.colorbar(cax, ax=ax)
        ax.plot(source_pixels[:, 1], source_pixels[:, 0], marker='*', markersize=3, color='w', ls='')
        ax.set_title("Smoothed Model After Source Subtraction", fontsize=10)
        ax.set_xlabel("Right Ascension", fontsize=10)

        fig.savefig(plot_outfile, dpi=150, bbox_inches='tight')

