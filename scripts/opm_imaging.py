#!/usr/bin/env python
"""
opm_imaging.py

Use CASA to make a multi-frequency synthesis image of a measurement set.
"""
from __future__ import print_function, division, absolute_import
import os
import argparse

# define argument parser
a = argparse.ArgumentParser(description="Run with: casa -c rtp_imaging.py <args>")
a.add_argument("--script", "-c", type=str, help="name of the script", required=True)
a.add_argument("--uvfitsname", type=str, required=True, help="name of measurement set to image")
a.add_argument("--image", type=str, required=True, help="name of output image")


def main():
    # parse arguments
    args = a.parse_args()

    # convert from uvfits to 

    # make sure we were given appropriate values
    visname = args.uvfitsname
    if not os.path.isfile(visname):
        raise IOError("{visname} does not exist".format(visname))
    visroot, visext = os.path.splitext(visname)
    if visext != ".uvfits":
        raise ValueError("visname must be a uvfits file ending in .uvfits")

    image = args.image
    imageroot, imageext = os.path.splitext(image)
    if imageext == ".ms":
        raise ValueError("image name cannot end in .ms")

    # convert from uvfits to measurement set
    msname = visroot + ".ms"
    print("converting uvfits to ms...")
    importuvfits(visname, msname)

    # call imaging command
    print("running CLEAN task...")
    clean(vis=msname, imagename=image, niter=0, weighting="briggs", robust=0,
          imsize=[512, 512], cell=["500 arcsec"], mode="mfs", nterms=1,
          spw="0:150~900", stokes="IQUV")

    vispolimname = visroot + ".vispol"
    clean(vis=msname, imagename=vispolimname, niter=0, weighting="briggs", robust=0,
          imsize=[512, 512], cell=["500 arcsec"], mode="mfs", nterms=1,
          spw="0:150~900", stokes="XXYY")

    return


if __name__ == "__main__":
    main()
