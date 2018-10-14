#!/usr/bin/env python2.7
"""
make_model_cube.py
==================

Given a series of MFS images, extract spectra of specified
sources, optionally smooth across freq, and insert into a proper CASA
image cube and write to disk.
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
from scipy.interpolate import interp1d
from scipy.signal import windows
from casa_imaging import casa_utils as utils
try:
    from sklearn import gaussian_process as gp
    gp_import = True
except:
    gp_import = False

pol2str = {-8: 'yx', -7: 'xy', -6: 'yy', -5: 'xx', -4: 'lr', -3: 'rl',
            -2: 'll', -1: 'rr', 1: 'I', 2: 'Q', 3: 'U', 4: 'V'}

args = argparse.ArgumentParser(description="Turn a series of MFS images into a spectral cube model.")

# IO Arguments
args.add_argument("imfiles", type=str, nargs='*', help="List of input MFS FITS images of fixed cell, imsize and polarization parameters. cell and imsize must be equal across both RA and Dec")
args.add_argument("--cubefile", type=str, help="Path to a proper CASA spectral cube FITS file with same cell, imsize and polarization parameters as input MFS images.")
args.add_argument("--sourcefile", type=str, help="File containing source locations to extract from MFS images. This should " \
                                                 "contain two tab-delimited columns holding RA and Dec respectively in degrees, " \
                                                 "and should be sorted by descending flux density.")
args.add_argument("--outfname", type=str, default=None, help="Output FITS filename of spectral cube model: Default is input cubefile.")
args.add_argument("--overwrite", default=False, action='store_true', help="overwrite output file.")
args.add_argument("--makeplots", default=False, action='store_true', help='Make plots of sources and their spectra.')
# Analysis Arguments
args.add_argument("--rb_Npix", default=41, type=int, help="Size of restoring beam cut-out in pixels, to use in source subtraction.")
args.add_argument("--fit_pl", default=False, action='store_true', help="Fit power law to spectra before GP smoothing. If GP smoothing, it operates on residual.")
args.add_argument("--fit_gp", default=False, action='store_true', help="Smooth spectra with a Gaussian process before linear interpolation.")
args.add_argument("--gp_ls", default=2.0, type=float, help="Smoothing lengthscale in MHz.")
args.add_argument("--gp_nl", default=0.1, type=float, help="GP noise level.")
args.add_argument("--gp_opt", default=None, type=str, help="GP optimizer. None is no optimization.")
args.add_argument("--gp_nrestarts", default=2, type=int, help="If optimizing, number of random state restarts.")
args.add_argument("--taper_alpha", default=0.1, type=float, help="Enact Tukey taper on smoothed spectra with specified alpha parameter: 0 is Tophat and 1 is Hanning.")
args.add_argument("--exclude_sources", default=[], type=int, nargs='*', help="Index of source(s) to exclude from MODEL.")

if __name__ == "__main__":

    # parse args
    a = args.parse_args()

    if not gp_import:
        args.fit_gp = False

    # check output
    if a.outfname is None:
        a.outfname = a.cubefile
    if os.path.exists(a.outfname) and not a.overwrite:
        raise IOError("Output file {} exists and overwrite is False, quitting...".format(a.outfname))

    # load spectral cube file
    chdu = fits.open(a.cubefile)
    chead = chdu[0].header
    cwcs = WCS(chead, naxis=2)

    # get hdu info
    cpols, cfreqs, cstokax, cfreqax = utils.get_hdu_info(chdu)
    Npols = len(cpols)

    # load sources
    src_ra, src_dec = np.loadtxt(a.sourcefile, delimiter='\t', usecols=(0, 1), dtype=np.float, unpack=True)

    # get source pixel locations
    ra_px, dec_px = np.around(cwcs.all_world2pix(src_ra, src_dec, 1), 0).astype(np.int)

    # iterate over imfiles
    im_cube = []
    rest_beams = []
    beam_widths = []
    freqs = []
    for imf in a.imfiles:
        # open hdu
        hdu = fits.open(imf)
        p, f, pa, fa = utils.get_hdu_info(hdu)

        # iterate over pols and get restoring beam
        rbs = []
        bws = []
        for i in range(len(cpols)):
            # get restoring beam info
            bmaj, bmin, bpa = utils.get_beam_info(hdu, pol_ind=i)
            bmajpx = np.abs(bmaj / hdu[0].header["CDELT1"])
            bminpx = np.abs(bmin / hdu[0].header["CDELT1"])

            # get restoring beam cut-out
            rbs.append(utils.make_restoring_beam(bmajpx, bminpx, bpa, size=a.rb_Npix))
            bws.append(np.mean([bmajpx, bminpx]))

        if np.isclose(bws, 0.0).any():
            continue

        # store data from zeroth frequency (b/c its an MFS image)
        im_cube.append(hdu[0].data[0])
        freqs.append(f)
        rest_beams.append(rbs)
        beam_widths.append(np.floor(bws).astype(np.int))

    im_cube = np.array(im_cube)
    rest_beams = np.array(rest_beams)
    beam_widths = np.array(beam_widths)
    freqs = np.array(freqs).ravel()

    # plot frequency slices with source masks
    if a.makeplots:
        # get output filename
        plotname = "{}.freqslices.png".format(os.path.splitext(a.outfname)[0])

        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        axes = axes.ravel()
        fig.subplots_adjust(hspace=0.15, wspace=0.1)
        loc = np.nanmedian(im_cube)
        scale = np.nanstd(im_cube)
        vmin = loc - scale
        vmax = loc + scale * 4

        Nthin = 1
        Nslices = len(im_cube)
        if Nslices > 9:
            Nthin = Nslices // 9
        for i in range(9):
            ax = axes[i]
            if i >= Nslices:
                ax.axis('off')
                continue
            if i >= len(im_cube[::Nthin]):
                continue
            im = im_cube[::Nthin][i]
            _im = np.nanmean(im, axis=0)
            f = freqs[::Nthin][i] / 1e6
            cax = ax.imshow(_im, origin='lower', vmin=vmin, vmax=vmax,
                           cmap='magma', aspect='auto')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title("{:.1f} MHz".format(f), fontsize=12, y=1)
            ax.scatter(ra_px, dec_px, facecolor='None', edgecolor='lime', s=50, lw=0.3)

        fig.subplots_adjust(right=0.90)
        cbax = fig.add_axes([0.90, 0.15, 0.05, 0.7])
        cbax.axis('off')
        cbar = fig.colorbar(cax, ax=cbax, fraction=0.75, aspect=40)

        fig.savefig(plotname, dpi=150, bbox_inches='tight')
        plt.close()

    # get spectrum of each source by taking peak flux in cut-out
    # and then subtracting peak flux convolved w/ restoring beam
    cutouts = []
    spectra = []
    for i, (ra, dec) in enumerate(zip(ra_px, dec_px)):
        freq_src = []
        freq_spc = []
        b = np.max(beam_widths)
        sc1 = slice(ra-b//2, ra+b//2+1)
        sc2 = slice(dec-b//2, dec+b//2+1)
        cutouts.append(np.nanmean(im_cube[:, :, sc2, sc1], axis=(0 ,1)))
        for j, f in enumerate(freqs):
            pol_src = []
            pol_spc = []
            for p in range(Npols):
                # carve out beam-shaped rectangle and take peak flux at each frequency
                b = beam_widths[j, p]
                sc1 = slice(ra-b//2, ra+b//2+1)
                sc2 = slice(dec-b//2, dec+b//2+1)
                src = im_cube[j, p, sc2, sc1]
                src_pk = np.nanmax(src)
                # get peak flux location and remove it convolved w/ restoring beam
                src_pk_px = np.ravel(np.where(src == src_pk)) + np.array([ra, dec]) - b//2
                sc1 = slice(src_pk_px[0]-a.rb_Npix//2, src_pk_px[0]+a.rb_Npix//2+1)
                sc2 = slice(src_pk_px[1]-a.rb_Npix//2, src_pk_px[1]+a.rb_Npix//2+1)
                im_cube[j, p, sc2, sc1] -= rest_beams[j, p] * 0.99 * src_pk
                pol_spc.append(src_pk)
            freq_spc.append(pol_spc)
        spectra.append(freq_spc)

    spectra = np.array(spectra)
    cutouts = np.array(cutouts)

    # plot postage cut-outs of sources
    if a.makeplots:
        # get output filename
        plotname = "{}.cutouts.png".format(os.path.splitext(a.outfname)[0])

        fig, axes = plt.subplots(5, 6, figsize=(10, 8))
        fig.subplots_adjust(hspace=0.2)
        axes = axes.ravel()
        for i, ax in enumerate(axes):
            if i >= len(spectra):
                ax.axis('off')
                continue
            cax = ax.imshow(np.log10(cutouts[i]), origin='lower', interpolation='nearest', cmap='nipy_spectral')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title("Source {}".format(i+1), fontsize=8, y=0.95)

        fig.savefig(plotname, dpi=150, bbox_inches='tight')
        plt.close()

    # plot source spectra before smoothing
    if a.makeplots:
        # get output name
        plotname = "{}.spectra.png".format(os.path.splitext(a.outfname)[0])

        fig, axes = plt.subplots(Npols, 1, figsize=(8, 6))
        fig.subplots_adjust(hspace=0.25)
        axes = np.asarray(axes)

        for i in range(Npols):
            ax = axes[i]
            for j, sp in enumerate(spectra):
                ax.plot(freqs/1e6, spectra[j, :, i], label=j+1, marker='o', ls='-', ms=4) 
            ax.grid()
            if i == 0:
                ax.legend(ncol=6, fontsize=6)
            ax.set_ylabel("Flux [Jy/Beam]", fontsize=10)
            ax.set_title("Source Spectra {} Polarization".format(pol2str[cpols[i]]), fontsize=10)
            if i == Npols-1:
                ax.set_xlabel("Frequency [MHz]", fontsize=10)

        fig.savefig(plotname, dpi=150, bbox_inches='tight')
        plt.close()

    # smooth spectra and interpolate to full frequency resolution of cube
    def smooth(x, y, x_out=None, fit_pl=True, fit_gp=True, ls=1.0, nl=0.1, n_restarts=1, optimizer=None, alpha=0):
        """ smooth input y array with power law and/or gaussian process
        x, y : input 1D x-array [MHz] and y-array
        x_out : output 1D x-array [MHz] to sample y-fit at
        fit_pl : fit a power law
        fit_gp : fit a gaussian process
        ls : gp length scale in MHz
        nl : gp noise level
        optimizer : optimizer to use in gp fitting: None is no optimization
        n_restarts : number of optimizer restarts
        """
        if fit_pl:
            # fit a power law and subtract it from y
            if np.any(y <= 0.0):
                pl = False
                fit = np.polyfit(x, y, 1)
                ypl = np.polyval(fit, x)
            else:
                pl = True
                fit = np.polyfit(np.log10(x), np.log10(y), 1)
                ypl = 10**np.polyval(fit, np.log10(x))
            y = y - ypl

        # make output x array and y array
        if x_out is None:
            x_out = x[:, None].copy()
        else:
            if x_out.ndim == 1:
                x_out = x_out[:, None]
        y_out = np.zeros_like(x_out.ravel())

        # GP smooth
        if fit_gp:
            # setup kernel
            kernel = 1**2 * gp.kernels.RBF(length_scale=ls, length_scale_bounds=(0.1, 1e3)) \
                     + gp.kernels.WhiteKernel(noise_level=nl, noise_level_bounds=(1e-5, 1e1))
            GP = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts,
                                             optimizer=optimizer)
            GP.fit(x[:, None], y)
            y_out += GP.predict(x_out).ravel()
            
        # Add power law back in
        if fit_pl:
            if pl:
                y_out += 10**np.polyval(fit, np.log10(x_out.ravel()))
            else:
                y_out += np.polyval(fit, x_out.ravel())

        # interpolate to full frequency resolution if no smoothing
        if not fit_pl and not fit_gp:
            y_out = interp1d(x, y, kind='linear', fill_value='extrapolate')(x_out.ravel())

        # add tapering to band edges
        y_out *= windows.tukey(len(x_out.ravel()), alpha)

        return y_out
         
    # iterate over spectra and smooth and/or interpolate to full freq resolution
    new_spectra = []
    for i, sp in enumerate(spectra):
        new_spec = []
        for j in range(Npols):
            if i in a.exclude_sources:
                # exclude certain sources from model
                result = np.zeros_like(cfreqs)
            else:
                result = smooth(freqs/1e6, sp[:, j], x_out=cfreqs/1e6, ls=a.gp_ls, nl=a.gp_nl, n_restarts=a.gp_nrestarts,
                                optimizer=a.gp_opt, fit_gp=a.fit_gp, fit_pl=a.fit_pl, alpha=a.taper_alpha)
            new_spec.append(result)
        new_spectra.append(new_spec)
    new_spectra = np.moveaxis(new_spectra, 1, 2)

    # plot spectra again
    if a.makeplots:
        Nsources = len(new_spectra)
        for p in range(Npols):
            plotname = "{}.smoothed_{}.png".format(os.path.splitext(a.outfname)[0], pol2str[cpols[p]])

            fig, axes = plt.subplots(int(np.ceil(Nsources / 4.0)), 4, figsize=(10, 12))
            Nv, Nh = axes.shape
            fig.subplots_adjust(hspace=0.2)
            ylim = new_spectra[:, :, p].min()-1, new_spectra[:, :, p].max()+1

            k = 0
            for i in range(Nv):
                for j in range(Nh):
                    ax = axes[i, j]
                    if k >= len(spectra):
                        ax.axis('off')
                        k += 1
                        continue
                    ax.grid()
                    ax.set_facecolor('lightgray')
                    p0, = ax.plot(freqs/1e6, spectra[k, :, p], lw=2, ls='', color='k', marker='.')
                    p1, = ax.plot(cfreqs/1e6, new_spectra[k, :, p], lw=2, ls='-', c='cyan')
                    ax.set_ylim(ylim)
                    ax.set_title("Source {}".format(k+1), fontsize=8, y=0.97)
                    if k == 3:
                        ax.legend([p0, p1], ['Raw', 'Smoothed'], fontsize=8)
                    if i == Nv-1:
                        ax.set_xlabel("Frequency [MHz]", fontsize=10)
                    else:
                        ax.set_xticklabels([])
                    if j == 0:
                        ax.set_ylabel("Flux [Jy/beam]", fontsize=8)
                    else:
                        ax.set_yticklabels([])
                    k += 1

            fig.savefig(plotname, dpi=150, bbox_inches='tight')
            plt.close()

    # make a new blank cube and insert source spectra
    new_cube = np.zeros_like(chdu[0].data, dtype=np.float64)
    for i, (ra, dec) in enumerate(zip(ra_px, dec_px)):
        new_cube[:, :, dec, ra] = new_spectra[i]

    # write new cube to file
    chdu[0].data = new_cube
    chdu.writeto(a.outfname, overwrite=True)

