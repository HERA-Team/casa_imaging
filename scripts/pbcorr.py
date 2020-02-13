#!/usr/bin/env python2.7
"""
pbcorr.py
=========

Primary Beam Correction
on FITS images, with a 
primary beam CST or healpix beam.

Nick Kern
July, 2019
nkern@berkeley.edu
"""
import numpy as np
import astropy.io.fits as fits
from astropy import wcs
from pyuvdata import UVBeam, utils as uvutils
import os
import sys
import glob
import argparse
import shutil
import copy
import healpy
import scipy.stats as stats
from casa_imaging import casa_utils
from scipy import interpolate
from astropy.time import Time
from astropy import coordinates as crd
from astropy import units as u
import warnings


args = argparse.ArgumentParser(description="Primary beam correction on FITS image files, given primary beam model")

args.add_argument("fitsfiles", type=str, nargs='*', help='path of image FITS file(s) to PB correct. assume all fits files have identical metadata except possibly for freq channel.')

# PB args
args.add_argument("--multiply", default=False, action='store_true', help='multiply data by primary beam, rather than divide')
args.add_argument("--lon", default=21.42830, type=float, help="longitude of observer in degrees east")
args.add_argument("--lat", default=-30.72152, type=float, help="latitude of observer in degrees north")
args.add_argument("--time", type=float, help='time of middle of observation in Julian Date')
args.add_argument("--image_x_orientation", default='east', type=str, help='x_orientation of fitsfiles, either ["east", "north"]. default is "east"')

# beam args
args.add_argument("--beamfile", type=str, help="path to primary beam in pyuvdata.uvbeam format", required=True)
args.add_argument("--pols", type=int, nargs='*', default=None, help="Polarization integer of healpix maps to use for beam models. Default is to use polarization in fits HEADER.")
args.add_argument("---freq_interp_kind", type=str, default='cubic', help="Interpolation method across frequency")

# IO args
args.add_argument("--ext", type=str, default="", help='Extension prefix for output file.')
args.add_argument("--outdir", type=str, default=None, help="output directory, default is path to fitsfile")
args.add_argument("--overwrite", default=False, action='store_true', help='overwrite output files')
args.add_argument("--silence", default=False, action='store_true', help='silence output to stdout')
args.add_argument("--spec_cube", default=False, action='store_true', help='assume all fitsfiles are identical except they each have a single but different frequency')

def echo(message, type=0):
    if verbose:
        if type == 0:
            print(message)
        elif type == 1:
            print('\n{}\n{}'.format(message, '-'*40))

if __name__ == "__main__":

    # parse args
    a = args.parse_args()
    verbose = a.silence == False

    # load pb
    echo("...loading beamfile {}".format(a.beamfile))
    # load beam
    uvb = UVBeam()
    uvb.read_beamfits(a.beamfile)
    if uvb.pixel_coordinate_system == 'healpix':
        uvb.interpolation_function = 'healpix_simple'
    else:
        uvb.interpolation_function = 'az_za_simple'
    uvb.freq_interp_kind = a.freq_interp_kind

    # get beam models and beam parameters
    beam_freqs = uvb.freq_array.squeeze() / 1e6
    Nbeam_freqs = len(beam_freqs)
    if uvb.x_orientation is None:
        # assume default is east
        warnings.warn("no x_orientation found in beam: assuming 'east' by default")
        uvb.x_orientation = 'east'
    beam_pols = [uvutils.polnum2str(p, x_orientation=uvb.x_orientation) for p in uvb.polarization_array]

    # iterate over FITS files
    for i, ffile in enumerate(a.fitsfiles):

        # create output filename
        if a.outdir is None:
            output_dir = os.path.dirname(ffile)
        else:
            output_dir = a.outdir

        output_fname = os.path.basename(ffile)
        output_fname = os.path.splitext(output_fname)
        if a.ext is not None:
            output_fname = output_fname[0] + '.pbcorr{}'.format(a.ext) + output_fname[1]
        else:
            output_fname = output_fname[0] + '.pbcorr' + output_fname[1]
        output_fname = os.path.join(output_dir, output_fname)

        # check for overwrite
        if os.path.exists(output_fname) and a.overwrite is False:
            raise IOError("{} exists, not overwriting".format(output_fname))

        # load hdu
        echo("...loading {}".format(ffile))
        hdu = fits.open(ffile)

        # get header and data
        head = hdu[0].header
        data = hdu[0].data

        # get polarization info
        ra, dec, pol_arr, data_freqs, stok_ax, freq_ax = casa_utils.get_hdu_info(hdu)
        Ndata_freqs = len(data_freqs)

        # get axes info
        npix1 = head["NAXIS1"]
        npix2 = head["NAXIS2"]
        nstok = head["NAXIS{}".format(stok_ax)]
        nfreq = head["NAXIS{}".format(freq_ax)]

        # replace with forced polarization if provided
        if a.pols is not None:
            pol_arr = np.asarray(a.pols, dtype=np.int)
        pols = [uvutils.polnum2str(pol, x_orientation=a.image_x_orientation) for pol in pol_arr]

        # make sure required pols exist in maps
        if not np.all([p in beam_pols for p in pols]):
            raise ValueError("Required polarizationns {} not all found in beam polarization array".format(pols))

        # convert from equatorial to spherical coordinates
        loc = crd.EarthLocation(lat=a.lat*u.degree, lon=a.lon*u.degree)
        time = Time(a.time, format='jd', scale='utc')
        equatorial = crd.SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='fk5', location=loc, obstime=time)
        altaz = equatorial.transform_to('altaz')
        theta = np.abs(altaz.alt.value - 90.0)
        phi = altaz.az.value

        # convert to radians
        theta *= np.pi / 180
        phi *= np.pi / 180

        if i == 0 or a.spec_cube is False:
            # evaluate primary beam
            echo("...evaluating PB")
            pb, _ = uvb.interp(phi.ravel(), theta.ravel(), polarizations=pols, reuse_spline=True)
            pb = np.abs(pb.reshape((len(pols), Nbeam_freqs) + phi.shape))

        # interpolate primary beam onto data frequencies
        echo("...interpolating PB")
        pb_shape = (pb.shape[1], pb.shape[2])
        pb_interp = interpolate.interp1d(beam_freqs, pb, axis=1, kind=a.freq_interp_kind, fill_value='extrapolate')(data_freqs / 1e6)

        # data shape is [naxis4, naxis3, naxis2, naxis1]
        if freq_ax == 4:
            pb_interp = np.moveaxis(pb_interp, 0, 1)

        # divide or multiply by primary beam
        if a.multiply is True:
            echo("...multiplying PB into image")
            data_pbcorr = data * pb_interp
        else:
            echo("...dividing PB into image")
            data_pbcorr = data / pb_interp

        # change polarization to interpolated beam pols
        head["CRVAL{}".format(stok_ax)] = pol_arr[0]
        if len(pol_arr) == 1:
            step = 1
        else:
            step = np.diff(pol_arr)[0]
        head["CDELT{}".format(stok_ax)] = step
        head["NAXIS{}".format(stok_ax)] = len(pol_arr)

        echo("...saving {}".format(output_fname))
        fits.writeto(output_fname, data_pbcorr, head, overwrite=True)

        output_pb = output_fname.replace(".pbcorr.", ".pb.")
        echo("...saving {}".format(output_pb))
        fits.writeto(output_pb, pb_interp, head, overwrite=True)

