"""
Tests for sky_cal.py.

Run from the command line as

    nosetests sky_cal.py

"""
import nose.tools as nt
import numpy as np
import os
import shutil
import glob
import copy
import sys
import subprocess
import unittest

## Note! In order to debug this script, you should open 
## a CASA session, and copy-paste each subprocess call
## into the session.

def test_source2file():
    out = subprocess.call(["../../scripts/source2file.py", "--ra", "30.05",
                           "--lon", "21.42", "--duration", "4", "--start_jd",
                           "2458101"])
    nt.assert_equal(out, 0)

def test_model_image():
    # clean up space
    sfiles = glob.glob("gleam*")
    for f in sfiles:
        try:
            os.remove(f)
        except:
            shutil.rmtree(f)
    # Test 1: Generate component list for 2 Hour Field            
    out = subprocess.call(["casa", "-c", "../../scripts/complist_gleam.py",
                           "--point_ra", "30.05", "--point_dec", "-30.72",
                           "--image", "--freqs", "148.828125,158.69140625,101",
                           "--imsize", "256", "--cell", "400arcsec",
                           "--min_flux", "0.1", "--gleamfile", "../data/small_gleam.fits",
                           "--radius", "15", "--overwrite"])
    nt.assert_equal(out, 0)
    nt.assert_true(np.all([os.path.exists("gleam_srcs.tab"), os.path.exists("gleam.cl"),
                           os.path.exists("gleam.cl.image"), os.path.exists("gleam.cl.fits")]))
    nt.assert_equal(np.loadtxt("gleam_srcs.tab", dtype=np.str, delimiter='\t').shape, (5612, 5))

    # Test 2: primary beam correction
    out = subprocess.call(["../../scripts/pbcorr.py", "--beamfile",
                           "../data/HERA_NF_dipole_power.beamfits",
                           "--lon", "21.42830", "--lat", "-30.72152", "--time", "2458101.29491",
                           "--pols", "-5", "-6", "--overwrite", "--multiply", "gleam.cl.fits"])

    # clean up space
    sfiles = glob.glob("gleam*")
    for f in sfiles:
        try:
            os.remove(f)
        except:
            shutil.rmtree(f)


def test_sky_cal():
    # Test 1: Basic Execution of Calibration
    # cleanup space
    sfiles = glob.glob("./skycal.ms*")
    for f in sfiles:
        try:
            os.remove(f)
        except:
            shutil.rmtree(f)
    shutil.copytree("../data/zen.2458101.28956.HH.uvRA.ms", "./skycal.ms")
    shutil.copy("../data/gleam02.loc", "./")
    # run sky_cal
    out = subprocess.call(["casa", "-c", "../../scripts/sky_cal.py",
                           "--msin", "./skycal.ms", "--source", "gleam02",
                           "--model", "../data/gleam02.cl.pbcorr.image",
                           "--refant", "53", "--KGcal", "--KGsnr", "0",
                           "--Acal", "--Asnr", "0", "--BPcal", "--gain_spw",
                           "0:0~100", "--bp_spw", "0:0~100", "--silence",
                           "--out_dir", "./"])
    nt.assert_equal(out, 0)
    nt.assert_true(np.all([os.path.exists('skycal.ms.K.cal'), os.path.exists('skycal.ms.K.cal.npz'),
                           os.path.exists('skycal.ms.Gphs.cal'), os.path.exists('skycal.ms.Gphs.cal.npz'),
                           os.path.exists('skycal.ms.Gamp.cal'), os.path.exists('skycal.ms.Gamp.cal.npz'),
                           os.path.exists('skycal.ms.B.cal'), os.path.exists('skycal.ms.B.cal.npz')]))
    # cleanup space
    sfiles = glob.glob("./skycal.ms*") + glob.glob("*.last") + glob.glob("*.log") + glob.glob("gleam*")
    for f in sfiles:
        try:
            os.remove(f)
        except:
            shutil.rmtree(f)


if __name__ == "__main__":
    unittest.main()
