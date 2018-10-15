"""
Utility functions for operating on CASA-exported FITS files.
"""
import astropy.io.fits as fits
import numpy as np
import os
import shutil
import scipy.stats as stats


def get_hdu_info(hdu):
    """
    Get info from a CASA-exported FITS header-unit list.

    Args:
        hdu : FITS header unit list
            Output from astropy.io.fits.open(<fname>)

    Returns: (pols, freqs, stokax, freqax)
        pols : ndarray containing polarization integers
        freqs : ndarray containing frequencies in Hz
        stokax : integer of polarization (stokes) axis in data cube
        freqax : integer of frequency axis in data cube
    """
    # get header
    head = hdu[0].header

    # get frequencies and polarizations
    if head["CTYPE3"] == "FREQ":
        freqax = 3
        stokax = 4
    elif head["CTYPE4"] == "FREQ":
        freqax = 4
        stokax = 3
    else:
        raise ValueError("Couldn't find freq and stokes axes in FITS file {}".format(hdu))

    # get pols
    pols = np.arange(head["NAXIS{}".format(stokax)]) * head["CDELT{}".format(stokax)] + head["CRVAL{}".format(stokax)]
    pols = np.asarray(pols, dtype=np.int)

    # get cube frequencies
    freqs = np.arange(head["NAXIS{}".format(freqax)]) * head["CDELT{}".format(freqax)] + head["CRVAL{}".format(freqax)]

    return pols, freqs, stokax, freqax


def get_beam_info(hdu, pol_ind=0, pxunits=False):
    """
    Takes a CASA-exported FITS HDU and gets the synthesized beam info
    in degrees. If pxunits, assumes CDELT1 and CDELT2 are equivalent.
    
    Args:
        hdu : FITS header unit list
            Output from astropy.io.fits.open(<fname>)
        pol_ind : integer
            Polarization index to query from beam information
        pxunits : boolean
            If True, return bmaj and bmin in pixel units

    Returns: (bmaj, bmin, bpa)
        bmaj : beam major axis in degrees (unless pxunits)
        bmin : beam minor axis in degrees (unless pxunits)
        bpa : beam position angle in degrees
    """
    # test for old clean output where its in the header in degrees
    if 'BMAJ' in hdu[0].header:
        bmaj = hdu[0].header['BMAJ']
        bmin = hdu[0].header['BMIN']
        bpa = hdu[0].header['BPA']
    else:
        # new tclean output where its in the data in arcsecs
        try:
            bmaj = hdu[1].data['BMAJ'][pol_ind] / 3600.
            bmin = hdu[1].data['BMIN'][pol_ind] / 3600.
            bpa = hdu[1].data['BPA'][pol_ind]
        except:
            raise ValueError("Couldn't get access to synthesized beam in HDU.")

    # convert bmaj and bmin to pixel units
    if pxunits:
        if not np.isclose(np.abs(hdu[0].header['CDELT1']), np.abs(hdu[0].header['CDELT2'])):
            raise ValueError("Can't convert to pixel units b/c CDELT1 != CDELT2, which this conversion assumes")
        bmaj = bmaj / np.abs(hdu[0].header['CDELT1'])
        bmin = bmin / np.abs(hdu[0].header['CDELT1'])

    return bmaj, bmin, bpa


def make_restoring_beam(bmaj, bmin, bpa, size=31):
    """
    Make a model of the restoring (clean) beam.

    Args:
        bmaj : beam major axis in pixel units
        bmin : beam minor axis in pixel units
        bpa : beam position angle in degrees
        size : integer side length of model in pixels. Must be odd.
    Returns:
        rest_beam : 2D ndarray of peak-normalized restoring beam
    """
    assert size % 2 == 1, "size must be odd-valued."
    # make a meshgrid
    x, y = np.meshgrid(np.linspace(size//2+1, -size//2, size), np.linspace(-size//2, size//2+1, size))
    P = np.array([x, y]).T
    # get bpa in radians and rotate meshgrid
    beam_theta = bpa * np.pi / 180
    Prot = P.dot(np.array([[np.cos(beam_theta), -np.sin(beam_theta)], [np.sin(beam_theta), np.cos(beam_theta)]]))
    # evaluate gaussian PDF, recall that its std is the major or minor axis / 2.0
    gauss_cov = np.array([[(bmaj/2.)**2, 0.0], [0.0, (bmin/2.)**2]])
    rest_beam = stats.multivariate_normal.pdf(Prot, mean=np.array([0, 0]), cov=gauss_cov)
    rest_beam /= rest_beam.max()

    return rest_beam


def subtract_beam(image, beam, px, peak_level=0.5, subtract=True, inplace=True):
    """
    Subtract a postage cutout of a synthesized beam from
    an image 2D array centered at image pixel values px, and
    read-off the peak flux in the cutout.

    Args:
        image : an nD image array with RA and Dec as 0th and 1st axes
        image : an nD beam array with RA and Dec as 0th and 1st axes.
            Must have the same CDELT as the image array.
        px : pixel coordinates of image to center subtraction at.
            Doesn't need to be within the image bounds.
        peak_level : beam level within which to look for peak flux
        subtract : bool, if True subtract the beam else add it.
        inplace : edit input array in memory, else make a copy

    Returns:
        diff_image : image with beam subtracted at px location
        peak : peak flux within peak_level
        im_cutout : cutout of image before subtraction
        bm_cutout : cutout of beam before subtraction
        im_s1, im_s2 : slice objects
    """
    # get slices
    beamNpx = beam.shape
    assert beamNpx[0] % 2 == 1 and beamNpx[1] % 2 == 1, "Beam must have odd-valued side-lengths"
    im_s1 = slice(px[0]-beamNpx[0]//2, px[0]+beamNpx[0]//2+1)
    im_s2 = slice(px[1]-beamNpx[1]//2, px[1]+beamNpx[1]//2+1)
    bm_s1 = slice(0, beamNpx[0])
    bm_s2 = slice(0, beamNpx[1])

    # confirm boundary values
    imNpx = image.shape
    if im_s1.start < 0:
        bm_s1 = slice(-im_s1.start, beamNpx[0])
        im_s1 = slice(0, im_s1.stop)
    if im_s1.stop > imNpx[0]:
        bm_s1 = slice(0, imNpx[0]-im_s1.stop)
        im_s1 = slice(im_s1.start, imNpx[0])
    if im_s2.start < 0:
        bm_s2 = slice(-im_s2.start, beamNpx[1])
        im_s2 = slice(0, im_s2.stop)
    if im_s2.stop > imNpx[1]:
        bm_s2 = slice(0, imNpx[1]-im_s2.stop)
        im_s2 = slice(im_s2.start, imNpx[1])

    # inplace
    if inplace:
        diff_image = image
    else:
        diff_image = image.copy()

    # get cutouts
    im_cutout = image[im_s1, im_s2].copy()
    bm_cutout = beam[bm_s1, bm_s2]

    def loop_peak(im, bm, plvl):
        if im.ndim > 2:
            pks = []
            sel = []
            for i in range(im.shape[2]):
                p, s = loop_peak(im[:, :, i], bm[:, :, i], plvl)
                pks.append(p)
                sel.append(s)
            return pks, sel
        else:
            s = bm > plvl
            if not s.max():
                return np.nan
            return np.nanmax(im[s]), s

    # look for peak flux within area defined by peak_level
    peak, select = loop_peak(im_cutout, bm_cutout, peak_level)
    if isinstance(peak, list):
        peak = np.array(peak)
    select = np.moveaxis(select, (0, 1), (-2, -1))

    # reformat bm_cutout given image dimensions
    if image.ndim > beam.ndim:
        bm_cutout = bm_cutout.reshape(bm_cutout.shape + tuple([1]*(image.ndim-beam.ndim)))

    # add peak value if beam is a float
    if np.issubclass_(bm_cutout.dtype.type, np.float):
        bm_cutout = bm_cutout * peak * 0.99999

    # difference
    if subtract:
        diff_image[im_s1, im_s2] -= bm_cutout
    else:
        diff_image[im_s1, im_s2] += bm_cutout

    return diff_image, peak, im_cutout, select, bm_cutout, im_s1, im_s2










