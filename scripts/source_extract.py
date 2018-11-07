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

try:
    from mpl_toolkits.mplot3d import Axes3D
    mplot = True
except:
    print("Could not import mpl_toolkits.mplot3d")
    mplot = False


a = argparse.ArgumentParser(description="Extract a source spectrum from a series of MFS images.")

a.add_argument("files", type=str, nargs='*', help="filename(s) or glob-parseable string of FITS filename(s)")
a.add_argument("--source", default=None, type=str, help="source name, with a <source>.loc file in working directory")
a.add_argument("--source_ext", default='', type=str, help="Extension after source name for output files.")
a.add_argument("--pols", default=1, type=int, nargs='*', help="Stokes polarization integer to extract. Default: 1")
a.add_argument("--radius", type=float, default=1, help="radius in degrees around estimated source position to get source peak")
a.add_argument("--rms_max_r", type=float, default=None, help="max radius in degrees around source to make rms calculation")
a.add_argument("--rms_min_r", type=float, default=None, help="min radius in degrees around source to make rms calculation")
a.add_argument("--outdir", type=str, default=None, help="output directory")
a.add_argument("--overwrite", default=False, action='store_true', help='overwite output')
a.add_argument("--gaussfit_mult", default=1.0, type=float, help="gaussian fit mask area is gaussfit_mult * synthesized_beam")
a.add_argument("--plot_fit", default=False, action='store_true', help="Make postage stamp images of the Gaussian fit performance.")

def source_extract(imfile, source=None, source_ext='', radius=1, gaussfit_mult=1.0,
                   rms_max_r=None, rms_min_r=None, pols=1, plot_fit=False):

    # open fits file
    hdu = fits.open(imfile)

    # get header
    head = hdu[0].header

    # determine if freq precedes stokes in header
    if head['CTYPE3'] == 'FREQ':
        freq_ax = 3
        stok_ax = 4
    else:
        freq_ax = 4
        stok_ax = 3

    # get axes info
    npix1 = head["NAXIS1"]
    npix2 = head["NAXIS2"]
    nstok = head["NAXIS{}".format(stok_ax)]
    nfreq = head["NAXIS{}".format(freq_ax)]

    # get frequency of image
    freq = head["CRVAL{}".format(freq_ax)]

    # get ra dec coordiantes
    ra_axis = np.linspace(head["CRVAL1"]-head["CDELT1"]*head["NAXIS1"]/2, head["CRVAL1"]+head["CDELT1"]*head["NAXIS1"]/2, head["NAXIS1"])
    dec_axis = np.linspace(head["CRVAL2"]-head["CDELT2"]*head["NAXIS2"]/2, head["CRVAL2"]+head["CDELT2"]*head["NAXIS2"]/2, head["NAXIS2"])
    RA, DEC = np.meshgrid(ra_axis, dec_axis)

    # get source coordinates
    if source is None:
        raise ValueError("Must specify a source with a <source>.loc file in working dir.")
    ra, dec = np.loadtxt('{}.loc'.format(source), dtype=str)
    ra, dec = map(float, ra.split(':')), map(float,dec.split(':'))
    ra = (ra[0] + ra[1]/60. + ra[2]/3600.) * 15
    dec = (dec[0] + np.sign(dec[0])*dec[1]/60. + np.sign(dec[0])*dec[2]/3600.)

    # get radius coordinates
    R = np.sqrt((RA - ra)**2 + (DEC - dec)**2)

    # select pixels
    select = R < radius

    # construct polarization array
    pol_arr = np.asarray(np.arange(nstok) * head["CDELT{}".format(stok_ax)] + head["CRVAL{}".format(stok_ax)], dtype=np.int)

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

        # check for clean file or tclean file
        if "BMAJ" in head:
            # clean output
            bmaj = head["BMAJ"]
            bmin = head["BMIN"]
            bpa = head["BPA"]
        else:
            # possibly tclean
            try:
                bmaj = hdu[1].data['BMAJ'][pol_ind] / 3600.
                bmin = hdu[1].data['BMIN'][pol_ind] / 3600.
                bpa = hdu[1].data["BPA"][pol_ind]
            except:
                raise ValueError("Couldn't find BMAJ or BMIN in hdu[0].header or hdu[1].data...")

        # check for tclean failed PSF
        if np.isclose(bmaj, bmin, 1e-6):
            raise ValueError("The PSF is not defined for pol {}.".format(polstr))

        # calculate beam area in degrees^2
        beam_area = (bmaj * bmin * np.pi / 4 / np.log(2))

        # calculate pixel area in degrees ^2
        pixel_area = np.abs(head["CDELT1"] * head["CDELT2"])
        Npix_beam = beam_area / pixel_area
        
        # get peak brightness within pixel radius
        _peak = np.nanmax(data[select])

        # get rms outside of pixel radius
        if rms_max_r is not None and rms_max_r is not None:
            rms_select = (R < rms_max_r) & (R > rms_min_r)
            _rms = np.sqrt(np.mean(data[select]**2))
        else:
            _rms = np.sqrt(np.mean(data[~select]**2))

        # get peak error
        _peak_err = _rms / np.sqrt(Npix_beam / 2.0)

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
        ecc = bmaj / bmin
        beam_theta = bpa * np.pi / 180 + np.pi/2
        EMAJ = R * np.sqrt(np.cos(T+beam_theta)**2 + ecc**2 * np.sin(T+beam_theta)**2)
        fit_mask = EMAJ < (bmaj / 2. * gaussfit_mult)
        masked_data = data.copy()
        masked_data[~fit_mask] = 0.0

        # fit 2d gaussian
        gauss_init = mod.functional_models.Gaussian2D(_peak, peak_ra, peak_dec, x_stddev=bmaj/2., y_stddev=bmin/2.) 
        fitter = mod.fitting.LevMarLSQFitter()
        gauss_fit = fitter(gauss_init, RA[fit_mask], DEC[fit_mask], data[fit_mask])

        # get gaussian fit properties
        _peak_gauss_flux = gauss_fit.amplitude.value
        P = np.array([X, Y]).T
        beam_theta -= np.pi/2
        Prot = P.dot(np.array([[np.cos(beam_theta), -np.sin(beam_theta)], [np.sin(beam_theta), np.cos(beam_theta)]]))
        gauss_cov = np.array([[gauss_fit.x_stddev.value**2, 0], [0, gauss_fit.y_stddev.value**2]])
        try:
            model_gauss = stats.multivariate_normal.pdf(Prot, mean=np.array([0,0]), cov=gauss_cov)
            model_gauss *= gauss_fit.amplitude.value / model_gauss.max()
            _int_gauss_flux = np.nansum(model_gauss.ravel()) / Npix_beam
        except:
            model_gauss = np.zeros_like(data)
            _int_gauss_flux = 0

        # append
        peak.append(_peak)
        peak_err.append(_peak_err)
        rms.append(_rms)
        peak_gauss_flux.append(_peak_gauss_flux)
        int_gauss_flux.append(_int_gauss_flux)

        # plot
        if plot_fit:
            # get postage cutout
            ra_select = np.where(np.abs(ra_axis-ra)<radius)[0]
            dec_select = np.where(np.abs(dec_axis-dec)<radius)[0]
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
    files = sorted(args.files)

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

    # iterate over files
    peak_flux = []
    peak_flux_err = []
    peak_gauss_flux = []
    int_gauss_flux = []
    freqs = []
    for i, fname in enumerate(files):
        try:
            output = source_extract(fname, **kwargs)
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
