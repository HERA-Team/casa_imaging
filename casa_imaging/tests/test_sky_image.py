"""
Tests for sky_image.py

Run from the command line as

    nosetests sky_image.py

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

def test_sky_image():
    # Test 1: Basic Execution of Imaging

    # clean up space
    sfiles = glob.glob("./skyimage.ms*")
    for f in sfiles:
        try:
            os.remove(f)
        except:
            shutil.rmtree(f)
    shutil.copytree("../data/zen.2458101.28956.HH.uvRA.ms", "./skyimage.ms")

    # First apply calibration solutions to data and split the model
    out = subprocess.call(["casa", "-c", "../../scripts/sky_cal.py",
                           "--msin", "./skyimage.ms",
                           "--gaintables", "../data/skycal.ms.K.cal",
                           "../data/skycal.ms.Gphs.cal",
                           "../data/skycal.ms.Gamp.cal",
                           "../data/skycal.ms.B.cal",
                           "--rflag",
                           "--model", "../data/gleam02.cl.pbcorr.image",
                           "--split_model"])
    nt.assert_true(os.path.exists('./skyimage.ms.model'))

    # Image the corrected data
    shutil.copy("../data/gleam02.loc", "./")
    out = subprocess.call(["casa", "-c", "../../scripts/sky_image.py",
                           "--msin", "./skyimage.ms", "--source", "gleam02",
                           "--out_dir", "./", "--image_mfs",
                           "--niter", "500", "500", "--imsize", "256", "--pxsize", "400",
                           "--stokes", "XXYY", "--weighting", "briggs", "--robust", "-1",
                           "--mask", "circle[[30.05deg,-30.7deg],5deg]", "",
                           "--deconvolver", "hogbom"])
    nt.assert_equal(out, 0)
    nt.assert_true(np.all([os.path.exists('skyimage.ms.gleam02.image'), os.path.exists('skyimage.ms.gleam02.image.fits'),
                           os.path.exists('skyimage.ms.gleam02.mask_1'), os.path.exists('skyimage.ms.gleam02.mask_2')]))

    # Image the model
    out = subprocess.call(["casa", "-c", "../../scripts/sky_image.py",
                           "--msin", "./skyimage.ms.model", "--source", "gleam02",
                           "--out_dir", "./", "--image_mfs",
                           "--niter", "500", "500", "--imsize", "256", "--pxsize", "400",
                           "--stokes", "XXYY", "--weighting", "briggs", "--robust", "-1",
                           "--mask", "circle[[30.05deg,-30.7deg],5deg]", "",
                           "--deconvolver", "hogbom"])
    nt.assert_equal(out, 0)

    # Image the residual
    out = subprocess.call(["casa", "-c", "../../scripts/sky_image.py",
                           "--msin", "./skyimage.ms", "--source", "gleam02",
                           "--source_ext", "_resid",
                           "--out_dir", "./", "--image_mfs", "--uvsub",
                           "--niter", "500", "500", "--imsize", "256", "--pxsize", "400",
                           "--stokes", "XXYY", "--weighting", "briggs", "--robust", "-1",
                           "--mask", "circle[[30.05deg,-30.7deg],5deg]", "",
                           "--deconvolver", "hogbom"])

    # Test 2: Make a diry MFS Spectral Cube: first reapply gains to corrected data
    out = subprocess.call(["casa", "-c", "../../scripts/sky_cal.py",
                           "--msin", "./skyimage.ms",
                           "--gaintables", "../data/skycal.ms.K.cal",
                           "../data/skycal.ms.Gphs.cal",
                           "../data/skycal.ms.Gamp.cal",
                           "../data/skycal.ms.B.cal",
                           "--rflag"])

    out = subprocess.call(["casa", "-c", "../../scripts/sky_image.py",
                           "--msin", "./skyimage.ms", "--source", "gleam02",
                           "--out_dir", "./", "--spec_cube", "--spec_dchan", "20",
                           "--spec_start", "0", "--spec_end", "100",
                           "--imsize", "256", "--pxsize", "400",
                           "--stokes", "I", "--weighting", "briggs", "--robust", "-1",
                           "--deconvolver", "hogbom"])

    # Run a Source Extraction
    out = subprocess.call(["../../scripts/source_extract.py"] + sorted(glob.glob("skyimage.ms.gleam02.spec????.image.fits")) \
                        + ["--source", "gleam02", "--radius", "2", "--pol", "1",
                           "--output_fname", "skyimage.ms.spec.tab", "--overwrite"])
    nt.assert_equal(np.loadtxt("skyimage.ms.spec.tab").shape, (5, 5))

    # clean up space
    sfiles = glob.glob("./skyimage.ms*") + glob.glob("gleam*") + glob.glob("*.last") + glob.glob("*.log")
    for f in sfiles:
        try:
            os.remove(f)
        except:
            shutil.rmtree(f)


if __name__ == "__main__":
    unittest.main()
