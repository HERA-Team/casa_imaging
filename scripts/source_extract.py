#!/usr/bin/env python2.7
"""
source_extract.py
========

get image statistics of FITS file(s)
on a specific source
"""
import astropy.io.fits as fits
from astropy import modeling as mod
import argparse
import os
import sys
import shutil
import glob
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import pyuvdata.utils as uvutils
import copy
from casa_imaging import casa_utils

try:
    from mpl_toolkits.mplot3d import Axes3D
    mplot = True
except:
    print("Could not import mpl_toolkits.mplot3d")
    mplot = False


a = argparse.ArgumentParser(description="Extract a source spectrum from a series of MFS images.")

a.add_argument("files", type=str, nargs='*', help="filename(s) or glob-parseable string of FITS filename(s)")
a.add_argument("--source", required=True, type=str, help="source name in the field")
a.add_argument("--source_ra", required=True, type=float, help="RA of source in J2000 degrees.")
a.add_argument("--source_dec", required=True, type=float, help="Dec of source in J2000 degrees.")
a.add_argument("--source_ext", default='', type=str, help="Extension after source name for output files.")
a.add_argument("--pols", default=1, type=int, nargs='*', help="Stokes polarization integer to extract. Default: 1")
a.add_argument("--radius", type=float, default=1, help="radius in degrees around estimated source position to get source peak")
a.add_argument("--rms_max_r", type=float, default=None, help="max radius in degrees around source to make rms calculation")
a.add_argument("--rms_min_r", type=float, default=None, help="min radius in degrees around source to make rms calculation")
a.add_argument("--outdir", type=str, default=None, help="output directory")
a.add_argument("--overwrite", default=False, action='store_true', help='overwite output')
a.add_argument("--gaussfit_mult", default=1.0, type=float, help="gaussian fit mask area is gaussfit_mult * synthesized_beam")
a.add_argument("--plot_fit", default=False, action='store_true', help="Make postage stamp images of the Gaussian fit performance.")

def source_extract(imfile, source, source_ra, source_dec, source_ext='', radius=1, gaussfit_mult=1.5,
                   rms_max_r=None, rms_min_r=None, pols=1, plot_fit=False):

    # open fits file
    hdu = fits.open(imfile)

    # get header
    head = hdu[0].header

    # get info
    RA, DEC, pol_arr, freqs, stok_ax, freq_ax = casa_utils.get_hdu_info(hdu)
    dra, ddec = head['CDELT1'], head['CDELT2']

    # get axes info
    npix1 = head["NAXIS1"]
    npix2 = head["NAXIS2"]
    nstok = head["NAXIS{}".format(stok_ax)]
    nfreq = head["NAXIS{}".format(freq_ax)]

    # get frequency of image
    freq = head["CRVAL{}".format(freq_ax)]

    # get radius coordinates: flat-sky approx
    R = np.sqrt((RA - source_ra)**2 + (DEC - source_dec)**2)

    # select pixels
    select = R < radius

    # polarization check
    if isinstance(pols, (int, np.integer, str, np.str)):
        pols = [pols]

    # iterate over polarizations
    peak, peak_err, rms, peak_gauss_flux, int_gauss_flux = [], [], [], [], []
    for pol in pols:
        # get polstr
        if isinstance(pol, (int, np.integer)):
            polint = pol
            polstr = uvutils.polnum2str(polint)
        elif isinstance(pol, (str, np.str)):
            polstr = pol
            polint = uvutils.polstr2num(polstr)

        if polint not in pol_arr:
            raise ValueError("Requested polarization {} not found in pol_arr {}".format(polint, pol_arr))
        pol_ind = pol_arr.tolist().index(polint)

        # get data
        if stok_ax == 3:
            data = hdu[0].data[0, pol_ind, :, :]
        elif stok_ax == 4:
            data = hdu[0].data[pol_ind, 0, :, :]

        # get beam info for this polarization
        bmaj, bmin, bpa = casa_utils.get_beam_info(hdu, pol_ind=pol_ind)

        # check for tclean failed PSF
        if np.isclose(bmaj, bmin, 1e-6):
            raise ValueError("The PSF is not defined for pol {}.".format(polstr))

        # relate FWHM of major and minor axes to standard deviation
        maj_std = bmaj / 2.35
        min_std = bmin / 2.35

        # calculate beam area in degrees^2
        # https://casa.nrao.edu/docs/CasaRef/image.fitcomponents.html
        beam_area = (bmaj * bmin * np.pi / 4 / np.log(2))

        # calculate pixel area in degrees^2
        pixel_area = np.abs(dra * ddec)
        Npix_beam = beam_area / pixel_area

        # get peak brightness within pixel radius
        _peak = np.nanmax(data[select])

        # get rms outside of source radius
        if rms_max_r is not None and rms_max_r is not None:
            rms_select = (R < rms_max_r) & (R > rms_min_r)
            _rms = np.sqrt(np.mean(data[rms_select]**2))
        else:
            _rms = np.sqrt(np.mean(data[~select]**2))

        ## fit a 2D gaussian and get integrated and peak flux statistics ##
        # recenter R array by peak flux point and get thata T array
        peak_ind = np.argmax(data[select])
        peak_ra = RA[select][peak_ind]
        peak_dec = DEC[select][peak_ind]
        X = (RA - peak_ra)
        Y = (DEC - peak_dec)
        R = np.sqrt(X**2 + Y**2)
        X[np.where(np.isclose(X, 0.0))] = 1e-5
        T = np.arctan(Y / X)

        # use synthesized beam as data mask
        ecc = maj_std / min_std
        beam_theta = bpa * np.pi / 180 + np.pi/2
        EMAJ = R * np.sqrt(np.cos(T+beam_theta)**2 + ecc**2 * np.sin(T+beam_theta)**2)
        fit_mask = EMAJ < (maj_std * gaussfit_mult)
        masked_data = data.copy()
        masked_data[~fit_mask] = 0.0

        # fit 2d gaussian
        gauss_init = mod.functional_models.Gaussian2D(_peak, peak_ra, peak_dec, x_stddev=maj_std, y_stddev=min_std) 
        fitter = mod.fitting.LevMarLSQFitter()
        gauss_fit = fitter(gauss_init, RA[fit_mask], DEC[fit_mask], data[fit_mask])

        # get gaussian fit properties
        _peak_gauss_flux = gauss_fit.amplitude.value
        P = np.array([X, Y]).T
        beam_theta -= np.pi/2  # correct for previous + np.pi/2
        Prot = P.dot(np.array([[np.cos(beam_theta), -np.sin(beam_theta)], [np.sin(beam_theta), np.cos(beam_theta)]]))
        gauss_cov = np.array([[gauss_fit.x_stddev.value**2, 0], [0, gauss_fit.y_stddev.value**2]])
        # try to get integrated flux
        try:
            model_gauss = stats.multivariate_normal.pdf(Prot, mean=np.array([0, 0]), cov=gauss_cov)
            model_gauss *= gauss_fit.amplitude.value / model_gauss.max()
            nanmask = ~np.isnan(model_gauss)
            _int_gauss_flux = np.nansum(model_gauss) / Npix_beam
        except:
            model_gauss = np.zeros_like(data)
            _int_gauss_flux = 0

        # get peak error
        # http://www.gb.nrao.edu/~bmason/pubs/m2mapspeed.pdf
        beam = np.exp(-((X / maj_std)**2 + (Y / min_std)**2))
        _peak_err = _rms / np.sqrt(np.sum(beam**2))

        # append
        peak.append(_peak)
        peak_err.append(_peak_err)
        rms.append(_rms)
        peak_gauss_flux.append(_peak_gauss_flux)
        int_gauss_flux.append(_int_gauss_flux)

        # plot
        if plot_fit:
            # get postage cutout
            ra_axis = RA[npix1//2]
            dec_axis = DEC[:, npix2//2]
            ra_select = np.where(np.abs(ra_axis-source_ra)<radius)[0]
            dec_select = np.where(np.abs(dec_axis-source_dec)<radius)[0]
            d = data[ra_select[0]:ra_select[-1]+1, dec_select[0]:dec_select[-1]+1]
            m = model_gauss[ra_select[0]:ra_select[-1]+1, dec_select[0]:dec_select[-1]+1]

            # setup wcs and figure
            wcs = WCS(head, naxis=2)
            fig = plt.figure(figsize=(14, 5))
            fig.subplots_adjust(wspace=0.2)
            fig.suptitle("Source {} from {}\n{:.2f} MHz".format(source, imfile, freq/1e6), fontsize=10)

            # make 3D plot
            if mplot:
                ax = fig.add_subplot(131, projection='3d')
                ax.axis('off')
                x, y = np.meshgrid(ra_select, dec_select)
                ax.plot_wireframe(x, y, m, color='steelblue', lw=2, rcount=20, ccount=20, alpha=0.75)
                ax.plot_surface(x, y, d, rcount=40, ccount=40, cmap='magma', alpha=0.5)

            # plot cut-out
            ax = fig.add_subplot(132, projection=wcs)
            cax = ax.imshow(data, origin='lower', cmap='magma')
            ax.contour(fit_mask, origin='lower', colors='lime', levels=[0.5])
            ax.contour(model_gauss, origin='lower', colors='snow', levels=np.array([0.5, 0.9]) * np.nanmax(m))
            ax.grid(color='w')
            cbar = fig.colorbar(cax, ax=ax)
            [tl.set_size(8) for tl in cbar.ax.yaxis.get_ticklabels()]
            [tl.set_size(10) for tl in ax.get_xticklabels()]
            [tl.set_size(10) for tl in ax.get_yticklabels()]
            ax.set_xlim(ra_select[0], ra_select[-1]+1)
            ax.set_ylim(dec_select[0], dec_select[-1]+1)
            ax.set_xlabel('Right Ascension', fontsize=12)
            ax.set_ylabel('Declination', fontsize=12)
            ax.set_title("Source Flux and Gaussian Fit", fontsize=10)

            # plot residual
            ax = fig.add_subplot(133, projection=wcs)
            resid = data - model_gauss
            vlim = np.abs(resid[fit_mask]).max()
            cax = ax.imshow(resid, origin='lower', cmap='magma', vmin=-vlim, vmax=vlim)
            ax.contour(fit_mask, origin='lower', colors='lime', levels=[0.5])
            ax.grid(color='w')
            ax.set_xlabel('Right Ascension', fontsize=12)
            cbar = fig.colorbar(cax, ax=ax)
            cbar.set_label(head['BUNIT'], fontsize=10)
            [tl.set_size(8) for tl in cbar.ax.yaxis.get_ticklabels()]
            [tl.set_size(10) for tl in ax.get_xticklabels()]
            [tl.set_size(10) for tl in ax.get_yticklabels()]
            ax.set_xlim(ra_select[0], ra_select[-1]+1)
            ax.set_ylim(dec_select[0], dec_select[-1]+1)
            ax.set_title("Residual", fontsize=10)

            fig.savefig('{}.{}.png'.format(os.path.splitext(imfile)[0], source + source_ext))
            plt.close()

    peak = np.asarray(peak)
    peak_err = np.asarray(peak_err)
    rms = np.asarray(rms)
    peak_gauss_flux = np.asarray(peak_gauss_flux)
    int_gauss_flux = np.asarray(int_gauss_flux)

    return peak, peak_err, rms, peak_gauss_flux, int_gauss_flux, freq

if __name__ == "__main__":

    # parse args
    args = a.parse_args()

    # sort files
    files = args.files

    # get filename
    if args.outdir is None:
        args.outdir = os.path.dirname(os.path.commonprefix(files))
    output_fname = "{}.{}{}.spectrum.npz".format(os.path.basename(os.path.splitext(os.path.commonprefix(files))[0]), args.source, args.source_ext)
    output_fname = os.path.join(args.outdir, output_fname)
    if os.path.exists(output_fname) and args.overwrite is False:
        raise IOError("file {} exists, not overwriting".format(output_fname))

    # get kwargs
    kwargs = copy.deepcopy(vars(args))
    del kwargs['files']
    del kwargs['source_ext']
    del kwargs['overwrite']
    del kwargs['outdir']
    source = kwargs['source']
    del kwargs['source']
    source_ra = kwargs['source_ra']
    del kwargs['source_ra']
    source_dec = kwargs['source_dec']
    del kwargs['source_dec']

    # iterate over files
    peak_flux = []
    peak_flux_err = []
    peak_gauss_flux = []
    int_gauss_flux = []
    freqs = []
    for i, fname in enumerate(files):
        try:
            output = source_extract(fname, source, source_ra, source_dec, **kwargs)
        except:
            print(sys.exc_info())
            continue
        peak_flux.append(output[0])
        peak_flux_err.append(output[2])
        peak_gauss_flux.append(output[3])
        int_gauss_flux.append(output[4])
        freqs.append(output[5])

    if len(peak_flux) == 0:
        raise ValueError("Couldn't get source flux from any input files")

    freqs = np.array(freqs)
    pols = np.asarray([pol if isinstance(pol, (int, np.integer)) else uvutils.polstr2num(pol) for pol in args.pols])
    peak_flux = np.array(peak_flux)
    peak_flux_err = np.array(peak_flux_err)
    peak_gauss_flux = np.array(peak_gauss_flux)
    int_gauss_flux = np.array(int_gauss_flux)

    # save spectrum
    print("...saving {}".format(output_fname))
    notes = "Freqs [MHz], Peak Flux [Jy/beam], Peak Gauss Flux [Jy/beam], Integrated Gauss Flux [Jy]"
    np.savez(output_fname, frequencies=freqs, polarizations=pols, peak_flux=peak_flux, peak_flux_err=peak_flux_err,
             peak_gauss_flux=peak_gauss_flux, integrated_gauss_flux=int_gauss_flux, notes=notes)
