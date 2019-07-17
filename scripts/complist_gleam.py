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

# IO Arguments
args.add_argument("--gleamfile", default="gleam.fits", type=str, help="Path to GLEAM point source catalogue FITS file [http://cdsarc.u-strasbg.fr/viz-bin/Cat?VIII/100].")
args.add_argument("--outdir", default='./', type=str, help='Output directory.')
args.add_argument("--ext", default='', type=str, help="Extension after 'gleam' for output files.")
args.add_argument("--overwrite", default=False, action='store_true', help="Overwrite output gleam.cl and gleam.im files.")

# Algorithm Arguments
args.add_argument("--point_ra", type=float, help="Pointing RA in degrees 0 < ra < 360.")
args.add_argument("--point_dec", type=float, help="Pointing Dec in degrees -90 < dec < 90.")
args.add_argument("--radius", type=float, default=5.0, help="Radius in degrees around pointing to get GLEAM sources.")
args.add_argument("--min_flux", default=0.0, type=float, help="Minimum flux at 151 MHz of sources to include in model.")
args.add_argument("--image", default=False, action='store_true', help='Make a FITS image of model')
args.add_argument("--freqs", default=None, type=str, help="Comma-separated values [MHz] for input into np.linspace({},{},{},endpoint=False)")
args.add_argument("--cell", default='200arcsec', type=str, help="Image pixel size in arcsec")
args.add_argument("--imsize", default=512, type=int, help="Image side-length in pixels.")
args.add_argument("--use_peak", default=False, action='store_true', help='Use peak flux rather than integrated flux in model.')
args.add_argument("--regions", default=None, type=str, help="Path to tab-delimited source file (see find_sources.py) that holds source RA and Dec in " \
                    "2nd and 3rd column respectively, within which to only include GLEAM point sources.")
args.add_argument("--region_radius", default=None, type=float, help="If providing a list of regions, this is the inclusion (exclusion) radius in degrees. B/c this uses flat-sky approx, only small-ish regions (few degrees at most) work well.")
args.add_argument("--exclude", default=False, action='store_true', help="If providing regions via --regions, exclude souces within masks, " \
                    "rather than only including sources within masks per default behavior.")
args.add_argument("--complists", type=str, nargs='*', default=None, help="Additional CASA component list strings or filepath to complist scripts.")

if __name__ == "__main__":
    a = args.parse_args()

    # check version:
    if '5.3.' in casa['version']:
        raise ValueError("Cannot run this script with CASA 5.3.* because it has a bug in its ia.modify() task!")

    basename = os.path.join(a.outdir, "gleam{}.cl".format(a.ext))

    if os.path.exists(basename) and not a.overwrite:
        print("{} already exists, not writing...".format(basename))
        sys.exit()

    # set pointing direction
    def deg2eq(ra, dec):
        _ra = ra / 15.0
        ra_h = int(np.floor(_ra))
        ra_m = int(np.floor((_ra - ra_h) * 60.))
        ra_s = int(np.around(((_ra - ra_h) * 60. - ra_m) * 60.))
        dec_d = int(np.floor(np.abs(dec)) * dec / np.abs(dec))
        dec_m = int(np.floor(np.abs(dec - dec_d) * 60.))
        dec_s = int(np.abs(dec - dec_d) * 3600. - dec_m * 60.)
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
        fstr = "Fp{:d}"
        fluxes = data['Fp151']
    else:
        fstr = "Fint{:d}"
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
    print("...a total of {} sources were found given RA, Dec and min_flux cuts".format(len(select)))

    # if regions provided, load them
    if a.regions is not None:
        mask_ra, mask_dec = np.loadtxt(a.regions, dtype=np.float, usecols=(2, 3), unpack=True)
        assert a.region_radius is not None, "if providing a list of sources, must specify region radius [deg]"
        mask_rad = a.region_radius

    # iterate over sources and add to complist
    select = select[np.argsort(dist[select])]
    for s in select:
        # get source info
        flux = fluxes[s]
        spix = data['alpha'][s]
        s_ra, s_dec = data['RAJ2000'][s], data['DEJ2000'][s]
        s_dir = deg2eq(s_ra, s_dec)
        name = "GLEAM {}".format(s_dir)

        # exclude or include if fed regions
        if a.regions is not None:
            mask_dists = np.sqrt((mask_ra - s_ra)**2 + (mask_dec - s_dec)**2)
            if a.exclude:
                if mask_dists.min() < mask_rad:
                    continue
            else:
                if mask_dists.min() >= mask_rad:
                    continue

        # if spectral index is a nan, try to derive it by hand
        if np.isnan(spix):
            frq = np.array([122., 130., 143., 151., 158., 166., 174.])
            x = []
            xstr = []
            for f in frq:
                xst = fstr.format(int(f))
                if xst in data.dtype.fields:
                    x.append(f)
                    xstr.append(xst)
            x = np.asarray(x, dtype=np.float)
            y = np.log10([data[s][xs] for xs in xstr])
            if sum(~np.isnan(y)) < 2:
                # skip this source b/c all but 1 bins are negative or nan...
                continue
            spix = np.polyfit(np.log10(x)[~np.isnan(y)], y[~np.isnan(y)], deg=1)[0]

            # if this is unreasonable, set to 0
            if spix < -3 or spix > 1:
                spix = 0

        # create component list
        cl.addcomponent(label=name, flux=flux, fluxunit="Jy", 
                        dir="J2000 {}".format(s_dir), freq=ref_freq, shape='point',
                        spectrumtype='spectral index', index=spix)

    # add other components if requested
    if a.complists is not None:
        for complist in a.complists:
            # first check if its a known file
            if os.path.exists(complist):
                with open(complist) as f:
                    exec(f.read())
            # otherwise interpret as a CASA Python string
            else:
                exec(complist)

    # iterate over sources and get metadata and append to list
    source = "{name:s}\t{flux:06.2f}\t{spix:02.2f}\t{ra:07.3f}\t{dec:07.3f}"
    sources = []
    for i in range(cl.length()):
        comp = cl.getcomponent(i)
        name = comp.get('label', None)
        flux = comp.get('flux', None).get('value', None)[0]
        spix = comp.get('spectrum', None).get('index', None)
        s_ra = comp.get('shape', None).get('direction', None).get('m0', None).get('value', None) * 180 / np.pi
        s_dec = comp.get('shape', None).get('direction', None).get('m1', None).get('value', None) * 180 / np.pi
        sources.append(source.format(name=name, flux=flux, spix=spix, ra=s_ra, dec=s_dec))

    # write source list to file
    print("...including {} sources".format(len(sources)))
    srcname = "{}.srcs.tab".format(basename)
    print("...saving {}".format(srcname))
    with open(srcname, "w") as f:
        f.write("# name\t flux [Jy]\t spix\t RA\t Dec\n")
        f.write('\n'.join(sources))

    # save
    print("...saving {}".format(basename))
    if os.path.exists(basename):
        shutil.rmtree(basename)
    cl.rename(basename)

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
        imname = "{}.image".format(basename)
        print("...saving {}".format(imname))
        ia.fromshape(imname, [a.imsize, a.imsize, 1, Nfreqs], overwrite=True)
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
        fitsname = "{}.fits".format(basename)
        print("...saving {}".format(fitsname))
        exportfits(imagename=imname, fitsimage=fitsname, overwrite=True, stokeslast=False)

    cl.close()

