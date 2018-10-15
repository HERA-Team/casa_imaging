"""
complist_gleam.py
-----------------

Script for making a low-frequency
CASA component list and/or FITS image
given the GLEAM point source catalogue.
http://cdsarc.u-strasbg.fr/viz-bin/Cat?VIII/100

Warning: Do not use CASA 5.3 to run this script, as it has a bug in ia.modify().

Nick Kern
Sept. 2018
nkern@berkeley.edu
"""
import os
import numpy as np
import argparse
import shutil
import sys
import pyfits

args = argparse.ArgumentParser(description="Run with casa as: casa -c complist_gleam.py <args>")
args.add_argument("-c", type=str, help="Name of this script")
args.add_argument("--point_ra", type=float, help="Pointing RA in degrees 0 < ra < 360.")
args.add_argument("--point_dec", type=float, help="Pointing Dec in degrees -90 < dec < 90.")
args.add_argument("--radius", type=float, default=5.0, help="Radius in degrees around pointing to get GLEAM sources.")
args.add_argument("--min_flux", default=0.0, type=float, help="Minimum flux at 151 MHz of sources to include in model.")
args.add_argument("--fill_spix", default=None, type=float, help="If spectral index doesn't exist for a source, use this. Default is to omit source.")
args.add_argument("--image", default=False, action='store_true', help='Make a FITS image of model')
args.add_argument("--freqs", default=None, type=str, help="Comma-separated values [MHz] for input into np.linspace({},{},{},endpoint=False)")
args.add_argument("--cell", default='200arcsec', type=str, help="Image pixel size in arcsec")
args.add_argument("--imsize", default=512, type=int, help="Image side-length in pixels.")
args.add_argument("--gleamfile", default="gleam.fits", type=str, help="Path to GLEAM point source catalogue FITS file [http://cdsarc.u-strasbg.fr/viz-bin/Cat?VIII/100].")
args.add_argument("--overwrite", default=False, action='store_true', help="Overwrite output gleam.cl and gleam.im files.")
args.add_argument("--use_peak", default=False, action='store_true', help='Use peak flux rather than integrated flux in model.')

if __name__ == "__main__":
    a = args.parse_args()

    # check version:
    if '5.3.' in casa['version']:
        raise ValueError("Cannot run this script with CASA 5.3.* because it has a bug in its ia.modify() task!")

    if os.path.exists("gleam.cl") and not a.overwrite:
        print("gleam.cl already exists, not writing...")
        sys.exit()

    # set pointing direction
    def deg2eq(ra, dec):
        _ra = ra / 15.0
        ra_h = int(np.floor(_ra))
        ra_m = int(np.floor((_ra - ra_h) * 60))
        ra_s = int(np.around(((_ra - ra_h) * 60 - ra_m) * 60))
        dec_d = int(np.floor(np.abs(dec)) * dec / np.abs(dec))
        dec_m = int(np.floor(np.abs(dec - dec_d) * 60.))
        dec_s = int(np.abs(dec - dec_d) * 3600 - dec_m * 60)
        direction = "{:02d}h{:02d}m{:02.0f}s {:03d}d{:02d}m{:02.0f}s".format(ra_h, ra_m, ra_s, dec_d, dec_m, dec_s)
        return direction

    direction = "J2000 {}".format(deg2eq(a.point_ra, a.point_dec))
    ref_freq = "151MHz"

    # Select all sources around pointing
    hdu = pyfits.open(a.gleamfile)
    data = hdu[1].data
    data_ra = data["RAJ2000"].copy()
    data_dec = data["DEJ2000"].copy()

    # get fluxes
    if a.use_peak:
        fluxes = data['Fp151']
    else:
        fluxes = data['Fint151']

    # correct for wrapping RA
    if a.point_ra < a.radius:
        data_ra[data_ra > a.point_ra + a.radius] -= 360.0
    elif np.abs(360.0 - a.point_ra) < a.radius:
        data_ra[data_ra < 360.0 - a.point_ra] += 360.0

    # select sources
    ra_dist = np.abs(data_ra - a.point_ra)
    dec_dist = np.abs(data_dec - a.point_dec)
    dist = np.sqrt(ra_dist**2 + dec_dist**2)
    select = np.where((dist <= a.radius) & (fluxes >= a.min_flux))[0]
    if len(select) == 0:
        raise ValueError("No sources found given RA, Dec and min_flux selections.")
    else:
        print("...including {} sources".format(len(select)))

    # iterate over sources and add to complist
    select = select[np.argsort(fluxes[select])[::-1]]
    source = "{name:s}\t{flux:06.2f}\t{spix:02.2f}\t{ra:07.3f}\t{dec:07.3f}"
    sources = []
    for s in select:
        flux = fluxes[s]
        spix = data['alpha'][s]
        s_ra, s_dec = data['RAJ2000'][s], data['DEJ2000'][s]
        if np.isnan(spix):
            if a.fill_spix is None:
                continue
            else:
                spix = a.fill_spix
        s_dir = deg2eq(s_ra, s_dec)
        name = "GLEAM {}".format(s_dir)
        cl.addcomponent(label=name, flux=flux, fluxunit="Jy", 
                        dir="J2000 {}".format(s_dir), freq=ref_freq, shape='point',
                        spectrumtype='spectral index', index=spix)
        sources.append(source.format(name=name, flux=flux, spix=spix, ra=s_ra, dec=s_dec))

    # write source list to file
    print("...saving gleam_srcs.tab")
    with open("gleam_srcs.tab", "w") as f:
        f.write("# name\t flux [Jy]\t spix\t RA\t Dec\n")
        f.write('\n'.join(sources))

    # save
    print("...saving gleam.cl")
    if os.path.exists("gleam.cl"):
        shutil.rmtree("gleam.cl")
    cl.rename("gleam.cl")

    # make image
    if a.image:
        # get frequencies
        if a.freqs is None:
            Nfreqs = 1
            freqs = np.array([151.0])
        else:
            freqs = np.linspace(*np.array(a.freqs.split(',')).astype(np.float), endpoint=False)
            Nfreqs = len(freqs)

        # setup image
        print("...saving gleam.cl.image")
        ia.fromshape("gleam.cl.image", [a.imsize, a.imsize, 1, Nfreqs], overwrite=True)
        cs = ia.coordsys()
        cs.setunits(['rad','rad','','Hz'])

        # set pixel properties
        cell_rad = qa.convert(qa.quantity(a.cell),"rad")['value']
        cs.setincrement([-cell_rad, cell_rad], type='direction')
        cs.setreferencevalue([qa.convert(direction.split()[1],'rad')['value'], qa.convert(direction.split()[2],'rad')['value']], type="direction")

        # set freq properties
        qa_freqs = qa.quantity(freqs, 'MHz')
        cs.setspectral(frequencies=qa_freqs)
 
        # set flux properties, make image, export to fits
        ia.setcoordsys(cs.torecord())
        ia.setbrightnessunit("Jy/pixel")
        ia.modify(cl.torecord(), subtract=False)
        print("...saving gleam.cl.fits")
        exportfits(imagename="gleam.cl.image", fitsimage="gleam.cl.fits", overwrite=True, stokeslast=False)

    cl.close()

