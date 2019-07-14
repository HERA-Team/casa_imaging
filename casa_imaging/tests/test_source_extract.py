"""
Test casa_imaging/scripts/source_extract.py
"""
import numpy as np
import sys
import casa_imaging
from casa_imaging.data import DATA_PATH
from astropy.io import fits
import shutil
import os
from scipy import stats, signal

# add scripts to path
sys.path.append(casa_imaging.SCRIPT_DIR)
from source_extract import source_extract

np.random.seed(0)

def setup():
    # load test image
    hdu = fits.open(os.path.join(DATA_PATH, "zen.2458101.28956.HH.uvR.CLEAN.image.fits"))

    # get info
    ra, dec, pols, freqs, sax, fax = casa_imaging.casa_utils.get_hdu_info(hdu)
    bmaj, bmin, bpa = casa_imaging.casa_utils.get_beam_info(hdu, pxunits=True)
    bpa *= -np.pi / 180

    # construct peak-normalized clean beam
    bmaj, bmin = bmaj / 2.35, bmin / 2.35  # correct for FWHM -> standard deviation
    x, y = np.linspace(-15, 15, 30), np.linspace(-15, 15, 30)
    X, Y = np.meshgrid(x, y)
    P = np.array([X, Y]).T
    Prot = P.dot(np.array([[np.cos(bpa), -np.sin(bpa)], [np.sin(bpa), np.cos(bpa)]]))
    gauss_cov = np.array([[(bmaj)**2, 0], [0, (bmin)**2]])
    model_gauss = stats.multivariate_normal.pdf(Prot, mean=np.array([0, 0]), cov=gauss_cov)
    model_gauss /= model_gauss.max()

    # clear data and insert a 18 Jy point source at center of image
    hdu[0].data[0, :] = 0.0
    hdu[0].data[0, 0, 256, 256] = 18.0
    hdu[0].data[0, 1, 258, 254] = 18.0  # make this polarization slightly offset

    # add some noise
    hdu[0].data += np.random.normal(0, 0.01, 512 * 512 * 2).reshape(1, 2, 512, 512)

    # convolve with clean beam
    hdu[0].data[0, 0] = signal.convolve2d(hdu[0].data[0, 0], model_gauss, mode='same', boundary='symm')
    hdu[0].data[0, 1] = signal.convolve2d(hdu[0].data[0, 1], model_gauss, mode='same', boundary='symm')

    # write to file
    fname = "./_test_im.fits"
    if os.path.exists(fname):
        os.remove(fname)
    hdu.writeto(fname, overwrite=True)

def test_source_extract():
    # load file
    setup()
    fname = "./_test_im.fits"
    source_ra = 30.133750
    source_dec = -30.97416

    # run source extraction
    (peak, peak_err, rms, peak_gauss_flux, int_gauss_flux,
     freq) = source_extract(fname, "test", source_ra, source_dec, radius=2.0, gaussfit_mult=2.0, pols=[-5, -6],
                            rms_min_r=1.0, rms_max_r=5.0)

    # ensure extracted flux in both pols is close to 18 Jy
    assert np.all(np.isclose(peak, 18.0, atol=1.0))
    assert np.all(np.isclose(peak_gauss_flux, 18.0, atol=1.0))
    assert np.all(np.isclose(int_gauss_flux, 18.0, atol=1.0))
    assert np.all(np.isclose(peak_err, 0.01, atol=1e-2))

    if os.path.exists(fname):
        os.remove(fname)
