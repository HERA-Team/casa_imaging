#!/usr/bin/env python2
"""
skycal_pipe.py
-----------------------------------------
This script is used as an automatic calibration
and imaging pipeline in CASA for HERA data.

See skycal_params.yml for relevant parameter selections.

Nicholas Kern
nkern@berkeley.edu
November, 2018
"""
import numpy as np
from pyuvdata import UVData
import pyuvdata.utils as uvutils
from casa_imaging import casa_utils as utils
import os
import sys
import glob
import yaml
from datetime import datetime
import json
import itertools
import shutil
from collections import OrderedDict as odict
from astropy.time import Time
import copy
import operator
import subprocess


#-------------------------------------------------------------------------------
# Parse YAML Configuration File
#-------------------------------------------------------------------------------
# Get config and load dictionary
config = sys.argv[1]
cf = utils.load_config(config)

# Consolidate IO, data and analysis parameter dictionaries
params = odict(cf['io'].items() + cf['data'].items() + cf['analysis'].items())
assert len(params) == len(cf['io']) + len(cf['data']) + len(cf['analysis']), ""\
       "Repeated parameters found within the scope of io, data and analysis dicts"

# Get algorithm dictionary
algs = cf['algorithm']
datafile = os.path.join(params['data_root'], params['data_file'])

# Get parameters used globally in the pipeline
verbose = params['verbose']
overwrite = params['overwrite']
casa = params['casa'].split() + params['casa_flags'].split()
point_ra = params['point_ra']
longitude = params['longitude']
latitude = params['latitude']
out_dir = params['out_dir']

# Change to working dir
os.chdir(params['work_dir'])

# Open a logfile
logfile = os.path.join(out_dir, params['logfile'])
if os.path.exists(logfile) and params['overwrite'] == False:
    raise IOError("logfile {} exists and overwrite == False, quitting pipeline...".format(logfile))
lf = open(logfile, "w")
if params['joinlog']:
    ef = lf
else:
    ef = open(os.path.join(params['out_dir'], params['errfile']), "w")
casa += ['--logfile', logfile]
sys.stdout = lf
sys.stderr = ef

# Setup (Small) Global Variable Dictionary
varlist = ['datafile', 'verbose', 'overwrite', 'out_dir', 'casa', 'point_ra', 'longitude',
           'latitude', 'lf', 'gaintables']
def global_vars(varlist=[]):
    d = []
    for v in varlist:
        try:
            d.append((v, globals()[v]))
        except KeyError:
            continue
    return dict(d)

# Print out parameter header
time = datetime.utcnow()
utils.log("Starting skycal_pipe.py on {}\n{}\n".format(time, '-'*60), 
             f=lf, verbose=verbose)
_cf = copy.copy(cf)
_cf.pop('algorithm')
utils.log(json.dumps(_cf, indent=1) + '\n', f=lf, verbose=verbose)

# Setup a dict->object converter
class Dict2Obj:
    def __init__(self, **entries):
        self.__dict__.update(entries)

#-------------------------------------------------------------------------------
# Search for a Source and Prepare Data for MS Conversion
#-------------------------------------------------------------------------------
if params['prep_data']:
    # start block
    time = datetime.utcnow()
    utils.log("\n{}\n...Starting PREP_DATA: {}\n".format("-"*60, time), 
                 f=lf, verbose=verbose)
    utils.log(json.dumps(algs['prep_data'], indent=1) + '\n', f=lf, verbose=verbose)
    p = Dict2Obj(**algs['prep_data'])

    # Check if datafile is already MS
    import source2file
    if os.path.splitext(datafile)[1] == '.ms':
        # get transit times
        (lst, transit_jd, utc_range, utc_center, source_files,
         source_utc_range) = source2file.source2file(point_ra, lon=longitude, lat=latitude,
                                                     duration=p.duration, start_jd=p.start_jd, get_filetimes=False,
                                                     verbose=verbose)
        utils.log("...the file {} is already a CASA MS, skipping rest of PREP_DATA".format(datafile), f=lf, verbose=verbose)
        timerange = utc_range

    else:
        # Iterate over polarizations
        if p.pols is None: p.pols = [None]
        uvds = []
        for pol in p.pols:
            if pol is None:
                pol = ''
            else:
                utils.log("...working on {} polarization".format(pol), f=lf, verbose=verbose)
                pol = '.{}.'.format(pol)

            # glob-parse the data file / template
            datafiles = [df for df in glob.glob(datafile) if pol in df]
            assert len(datafiles) > 0, "Searching for {} with pol {} but found no files...".format(datafile, pol)

            # get transit times
            import source2file
            (lst, transit_jd, utc_range, utc_center, source_files,
             source_utc_range) = source2file.source2file(point_ra, lon=longitude, lat=latitude,
                                                         duration=p.duration, start_jd=p.start_jd, get_filetimes=p.get_filetimes,
                                                         verbose=verbose, jd_files=copy.copy(datafiles))
            timerange = utc_range

            # ensure source_utc_range and utc_range are similar
            if source_utc_range is not None:
                utc_range_start = utc_range.split('~')[0].strip('"').split('/')
                utc_range_start = map(int, utc_range_start[:-1] + utc_range_start[-1].split(':'))
                utc_range_start = Time(datetime(*utc_range_start), format='datetime').jd
                source_utc_range_start = source_utc_range.split('~')[0].strip('"').split('/')
                source_utc_range_start = map(int, source_utc_range_start[:-1] + source_utc_range_start[-1].split(':'))
                source_utc_range_start = Time(datetime(*source_utc_range_start), format='datetime').jd
                # if difference is larger than 1 minute,
                # then probably the correct files were not found
                if np.abs(utc_range_start - source_utc_range_start) * 24 * 60 > 1:
                    utils.log("Warning: Difference between theoretical transit time and transit time " \
                        "deduced from files found is larger than 1-minute: probably the correct " \
                        "files were not found because the correct files did not exist under the " \
                        "data template {}".format(datafile), f=lf, verbose=verbose)
                timerange = source_utc_range

            # load data into UVData
            utils.log("...loading data", f=lf, verbose=verbose)
            _uvds = []
            for sf in list(source_files):
                # read data
                _uvd = UVData()
                _uvd.read(sf, antenna_nums=p.antenna_nums)

                # read flagfile if fed
                if p.flag_ext != "":
                    flagfile = glob.glob("{}{}".format(sf, p.flag_ext))
                    if len(flagfile) == 1:
                        utils.log("...loading and applying flags {}".format(flagfile[0]), f=lf, verbose=verbose)
                        ff = np.load(flagfile[0])
                        _uvd.flag_array += ff['flag_array']

                # append to list
                _uvds.append(_uvd)

            # concatenate source files
            uvd = reduce(operator.add, _uvds)

            # isolate only relevant times
            times = np.unique(uvd.time_array)
            times = times[np.abs(times-transit_jd) < (p.duration / (24. * 60. * 2))]
            assert len(times) > 0, "No times found in source_files {} given transit JD {} and duration {}".format(source_files, transit_jd, p.duration)
            uvd.select(times=times)

            # append
            uvds.append(uvd)

        # concatenate uvds
        uvd = reduce(operator.add, uvds)

        # get output filepath w/o uvfits extension if provided
        outfile = os.path.join(params['out_dir'], p.outfile.format(uvd.time_array.min()))
        if os.path.splitext(outfile)[1] == '.uvfits':
            outfile = os.path.splitext(outfile)[0]

        # write to file
        if uvd.phase_type == 'phased':
            # write uvfits
            uvfits_outfile = outfile + '.uvfits'
            if not os.path.exists(uvfits_outfile) or overwrite:
                utils.log("...writing {}".format(uvfits_outfile), f=lf, verbose=verbose)
                uvd.write_uvfits(uvfits_outfile, spoof_nonessential=True)
            # unphase to drift
            uvd.unphase_to_drift()
            # write miriad
            if not os.path.exists(outfile) or overwrite:
                utils.log("...writing {}".format(outfile), f=lf, verbose=verbose)
                uvd.write_miriad(outfile, clobber=True)
        elif uvd.phase_type == 'drift':
            # write miriad
            if not os.path.exists(outfile) or overwrite:
                utils.log("...writing {}".format(outfile), f=lf, verbose=verbose)
                uvd.write_miriad(outfile, clobber=True)
            # write uvfits
            uvfits_outfile = outfile + '.uvfits'
            if not os.path.exists(uvfits_outfile) or overwrite:
                uvd.phase_to_time(Time(transit_jd, format='jd'))
                utils.log("...writing {}".format(uvfits_outfile), f=lf, verbose=verbose)
                uvd.write_uvfits(uvfits_outfile, spoof_nonessential=True)

        # convert the uvfits file to ms
        ms_outfile = outfile + '.ms'
        utils.log("...converting to Measurement Set")
        if not os.path.exists(ms_outfile) or overwrite:
            if os.path.exists(ms_outfile):
                shutil.rmtree(ms_outfile)
            utils.log("...writing {}".format(ms_outfile), f=lf, verbose=verbose)
            ecode = subprocess.check_call(casa + ["-c", "importuvfits('{}', '{}')".format(uvfits_outfile, ms_outfile)])

        # overwrite relevant parameters for downstream analysis
        datafile = ms_outfile

        del uvds, uvd

    # overwrite downstream parameters
    algs['gen_model']['time'] = transit_jd

    # end block
    time2 = datetime.utcnow()
    utils.log("...finished PREP_DATA: {:d} sec elapsed".format(utils.get_elapsed_time(time, time2)), f=lf, verbose=verbose)

#-------------------------------------------------------------------------------
# Generate Flux Model
#-------------------------------------------------------------------------------

# Make a Model Generation Function
def gen_model(**kwargs):
    p = Dict2Obj(**kwargs)
    utils.log("\n{}\n...Generating a Flux Model", f=p.lf, verbose=p.verbose)

    # compile complist_gleam.py command
    cmd = casa + ["-c", "complist_gleam.py"]
    cmd += ['--point_ra', p.point_ra, '--point_dec', p.latitude, '--outdir', p.out_dir, 
            '--gleamfile', p.gleamfile, '--radius', p.radius, '--min_flux', p.min_flux,
            '--freqs', p.freqs, '--cell', p.cell, '--imsize', p.imsize]
    if p.image:
        cmd += ['--image']
    if p.use_peak:
        cmd += ['--use_peak']
    if p.overwrite:
        cmd += ['--overwrite']
    if hasattr(p, 'regions'):
        cmd += ['--regions', p.regions, '--exclude', '--region_radius', p.region_radius]
    if hasattr(p, 'file_ext'):
        cmd += ['--ext', p.file_ext]
    else:
        p.file_ext = ''
    cmd = map(str, cmd)
    ecode = subprocess.check_call(cmd)

    modelstem = "gleam{}.cl".format(p.file_ext)
    model = os.path.join(p.out_dir, modelstem)
    if p.image:
        model += ".image"

    # pbcorrect
    if p.pbcorr:
        utils.log("...applying PB to model", f=p.lf, verbose=p.verbose)
        assert p.image, "Cannot pbcorrect flux model without image == True"
        cmd = ["pbcorr.py", "--lon", p.longitude, "--lat", p.latitude, "--time", p.time, "--pols"] \
               + [uvutils.polstr2num(pol) for pol in p.pols] \
               + ["--outdir", p.out_dir, "--multiply", "--beamfile", p.beamfile]
        if p.overwrite:
            cmd.append("--overwrite")
        cmd.append(modelstem + '.fits')

        # generate component list and / or image cube flux model
        cmd = map(str, cmd)
        ecode = subprocess.check_call(cmd)

        # importfits
        cmd = p.casa + ["-c", "importfits('{}', '{}', overwrite={})".format(modelstem + '.pbcorr.fits', modelstem + '.pbcorr.image', p.overwrite)]
        ecode = subprocess.check_call(cmd)
        model = modelstem + ".pbcorr.image"

    return model

if params['gen_model']:
    # start block
    time = datetime.utcnow()
    utils.log("\n{}\n...Starting GEN_MODEL: {}\n".format("-"*60, time), 
                 f=lf, verbose=verbose)
    utils.log(json.dumps(algs['gen_model'], indent=1) + '\n', f=lf, verbose=verbose)

    # Generate Model
    model = gen_model(**dict(algs['gen_model'].items() + global_vars(varlist).items()))

    # update di_cal model path
    algs['di_cal']['model'] = model

    # end block
    time2 = datetime.utcnow()
    utils.log("...finished GEN_MODEL: {:d} sec elapsed".format(utils.get_elapsed_time(time, time2)), f=lf, verbose=verbose)

#-------------------------------------------------------------------------------
# Direction Independent Calibration
#-------------------------------------------------------------------------------

# Define a calibration Function
def calibrate(**kwargs):
    p = Dict2Obj(**kwargs)
    # If source.loc file doesn't exist, write it
    if not os.path.exists("{}.loc".format(p.source)):
        direction = utils.get_direction(p.source_ra, p.source_dec)
        with open('{}.loc'.format(p.source), 'w') as f:
            f.write(direction)

    # compile command
    cmd = p.casa + ["-c", "sky_cal.py"]
    cmd += ["--msin", p.datafile, "--source", p.source, "--out_dir", p.out_dir, "--model", p.model,
            "--refant", p.refant, "--gain_spw", p.gain_spw, "--uvrange", p.uvrange, "--timerange",
            p.timerange, "--ex_ants", p.ex_ants, "--gain_ext", p.gain_ext, '--bp_spw', p.bp_spw]
    if isinstance(p.gaintables, list):
        gtables = p.gaintables
    else:
        if p.gaintables in ['', None, 'None', 'none']:
            gtables = []
        else:
            gtables = [p.gaintables]
    if len(gtables) > 0:
        cmd += ['--gaintables'] + gtables
    if p.rflag:
        cmd += ["--rflag"]
    if p.KGcal:
        cmd += ["--KGcal", "--KGsnr", p.KGsnr]
    if p.Acal:
        cmd += ["--Acal", "--Asnr", p.Asnr]
    if p.BPcal:
        cmd += ["--BPcal", "--BPsnr", p.BPsnr, "--bp_spw", p.bp_spw]
    if p.BPsolnorm:
        cmd += ['--BPsolnorm']
    if p.split_cal:
        cmd += ["--split_cal", "--cal_ext", p.cal_ext]
    if p.split_model:
        cmd += ["--split_model"]
    cmd = [' '.join(_cmd) if type(_cmd) == list else str(_cmd) for _cmd in cmd]

    utils.log("...starting calibration", f=p.lf, verbose=p.verbose)
    ecode = subprocess.check_call(cmd)

    # Gather gaintables
    gext = ''
    if p.gain_ext not in ['', None]:
        gext = '.{}'.format(p.gain_ext)
    gts = sorted(glob.glob("{}{}.?.cal".format(p.datafile, gext)) + glob.glob("{}{}.????.cal".format(p.datafile, gext)))

    # export to calfits if desired
    if p.export_gains:
        utils.log("...exporting\n{}\n to calfits and combining into a single cal table".format('\n'.join(gts)), f=p.lf, verbose=p.verbose)
        # do file checks
        mirvis = os.path.splitext(p.datafile)[0]
        gtsnpz = ["{}.npz".format(gt) for gt in gts]
        if not os.path.exists(mirvis):
            utils.log("...{} doesn't exist: cannot export gains to calfits".format(mirvis), f=p.lf, verbose=p.verbose)

        elif len(gts) == 0:
            utils.log("...no gaintables found, cannot export gains to calfits", f=p.lf, verbose=p.verbose)

        elif not np.all([os.path.exists(gtnpz) for gtnpz in gtsnpz]):
            utils.log("...couldn't find a .npz file for all input .cal tables, can't export to calfits", f=p.lf, verbose=p.verbose)

        else:
            calfits_fname = "{}.{}{}.calfits".format(mirvis, p.source, p.gain_ext)
            cmd = ['skynpz2calfits.py', "--fname", calfits_fname, "--uv_file", mirvis, '--out_dir', p.out_dir]
            if p.overwrite:
                cmd += ["--overwrite"]
            # add a delay and phase solution
            matchK = ["K.cal.npz" in gt for gt in gtsnpz]
            matchGphs = ["Gphs.cal.npz" in gt for gt in gtsnpz]
            if np.any(matchK):
                cmd += ["--plot_dlys"]
                cmd += ["--dly_files"] + [gtsnpz[i] for i, b in enumerate(matchK) if b == True]
                if not np.any(matchGphs):
                    utils.log("...WARNING: A delay file {} was found, but no mean phase file, which is needed if a delay file is present.", f=lf, verbose=verbose)
            if np.any(matchGphs):
                cmd += ["--plot_phs"]
                cmd += ["--phs_files"] + [gtsnpz[i] for i, b in enumerate(matchGphs) if b == True]

            # add a mean amp solution
            matchGamp = ["Gamp.cal.npz" in gt for gt in gtsnpz]
            if np.any(matchGamp):
                cmd += ["--plot_amp"]
                cmd += ["--amp_files"] + [gtsnpz[i] for i, b in enumerate(matchGamp) if b == True]

            # add a bandpass solution
            matchB = ["B.cal.npz" in gt for gt in gtsnpz]
            if np.any(matchB):
                cmd += ["--plot_bp"]
                cmd += ["--bp_files"] + [gtsnpz[i] for i, b in enumerate(matchB) if b == True]

            # additional smoothing options
            if p.smooth:
                cmd += ["--bp_gp_smooth", "--bp_gp_max_dly", p.gp_max_dly]
            if p.medfilt:
                cmd += ["--bp_medfilt", "--medfilt_kernel", p.kernel]
            if p.bp_broad_flags:
                cmd += ["--bp_broad_flags", "--bp_flag_frac", p.bp_flag_frac]
            if not p.verbose:
                cmd += ['--silence']

            cmd = map(str, cmd)
            ecode = subprocess.check_call(cmd)

            # convert calfits back to a single Btotal.cal table
            if np.any(matchB):
                # convert to cal
                bfile = gts[matchB.index(True)]
                btot_file = os.path.join(out_dir, "{}{}.Btotal.cal".format(os.path.basename(p.datafile), gext))
                cmd = p.casa + ["-c", "calfits_to_Bcal.py", "--cfits", calfits_fname, "--inp_cfile", bfile,"--out_cfile", btot_file]
                if overwrite:
                    cmd += ["--overwrite"]
                ecode = subprocess.check_call(cmd)
                # replace gaintables with Btotal.cal
                gts = [btot_file]

    # append to gaintables
    gtables += gts

    return gtables

# Define an Imaging Function
def image(**kwargs):
    p = Dict2Obj(**kwargs)
    # compile general image command
    if p.image_mfs or p.image_spec:
        cmd = p.casa + ["-c", "sky_image.py"]
        cmd += ["--source", p.source, "--out_dir", p.out_dir,
                "--pxsize", p.pxsize, "--imsize", p.imsize,
                "--uvrange", p.uvrange, "--timerange", p.timerange,
                "--stokes", p.stokes, "--weighting", p.weighting, "--robust", p.robust,
                "--pblimit", p.pblimit, "--deconvolver", p.deconvolver, "--niter",
                p.niter, '--cycleniter', p.cycleniter, '--threshold', p.threshold,
                '--mask', p.mask]
        cmd = [map(str, _cmd) if type(_cmd) == list else str(_cmd) for _cmd in cmd]
        cmd = reduce(operator.add, [i if type(i) == list else [i] for i in cmd])

    # Perform MFS imaging
    if p.image_mfs:
        # Image Corrected Data
        utils.log("...starting MFS image of CORRECTED data", f=p.lf, verbose=p.verbose)
        icmd = cmd + ['--image_mfs', '--msin', p.datafile, "--source_ext", p.source_ext, '--spw', p.spw] 
        ecode = subprocess.check_call(icmd)

        # Image the split MODEL
        if p.image_mdl:
            utils.log("...starting MFS image of MODEL data", f=p.lf, verbose=p.verbose)
            mfile = "{}.model".format(p.datafile)
            if not os.path.exists(mfile):
                utils.log("Didn't split model from datafile, which is required to image the model", f=p.lf, verbose=p.verbose)
            else:
                icmd = cmd + ['--image_mfs', '--msin', mfile, '--source_ext', p.source_ext]
                ecode = subprocess.check_call(icmd)

        # Image the CORRECTED - MODEL residual
        if p.image_res:
            utils.log("...starting MFS image of CORRECTED - MODEL data", f=p.lf, verbose=p.verbose)
            icmd = cmd + ['--image_mfs', '--msin', p.datafile, '--uvsub', "--source_ext", "{}_resid".format(p.source_ext)] 
            ecode = subprocess.check_call(icmd)

            # Apply gaintables to make CORRECTED column as it was
            utils.log("...reapplying gaintables to CORRECTED data", f=p.lf, verbose=p.verbose)
            cmd2 = p.casa + ["-c", "sky_cal.py", "--msin", p.datafile, "--gaintables"] + p.gaintables
            ecode = subprocess.check_call(cmd2)

    if p.image_spec:
        # Perform Spectral Cube imaging
        utils.log("...starting spectral cube imaging", f=p.lf, verbose=p.verbose)
        icmd = cmd + ['--spec_cube', '--msin', p.datafile, "--source_ext", p.source_ext,
                      '--spec_start', str(p.spec_start), '--spec_end', str(p.spec_end),
                      '--spec_dchan', str(p.spec_dchan)] 
        ecode = subprocess.check_call(icmd)

        # Collate output images and Run a Source Extraction
        img_cube = sorted(glob.glob("{}.{}{}.spec????.image.fits".format(p.datafile, p.source, p.source_ext)))
        if p.source_extract:
            utils.log("...extracting {} source spectra".format(p.source), f=p.lf, verbose=p.verbose)
            if len(img_cube) == 0:
                utils.log("...no image cube files found, cannot extract spectrum", f=p.lf, verbose=p.verbose)
            else:
                cmd = ["source_extract.py", "--source", p.source, "--radius", p.radius, '--pols'] \
                      + p.pols + ["--outdir", p.out_dir, "--gaussfit_mult", p.gauss_mult]
                if p.overwrite:
                    cmd += ["--overwrite"]
                if p.plot_fit:
                    cmd += ["--plot_fit"]
                cmd += img_cube
                cmd = map(str, cmd)
                ecode = subprocess.check_call(cmd)


# Start Calibration
if params['di_cal']:
    # start block
    time = datetime.utcnow()
    utils.log("\n{}\n...Starting DI_CAL: {}\n".format("-"*60, time), f=lf, verbose=verbose)
    cal_kwargs = dict(algs['gen_cal'].items() + algs['di_cal'].items())
    utils.log(json.dumps(cal_kwargs, indent=1) + '\n', f=lf, verbose=verbose)
    gaintables = cal_kwargs['gaintables']

    # Perform Calibration
    gaintables = calibrate(**dict(cal_kwargs.items() + global_vars(varlist).items()))

    # Perform Imaging
    image(**dict(cal_kwargs.items() + global_vars(varlist).items()))

    # end block
    time2 = datetime.utcnow()
    utils.log("...finished DI_CAL: {:d} sec elapsed".format(utils.get_elapsed_time(time, time2)), f=lf, verbose=verbose)

#-------------------------------------------------------------------------------
# Direction Dependent Calibration
#-------------------------------------------------------------------------------
if params['dd_cal']:
    # start block
    time = datetime.utcnow()
    utils.log("\n{}\n...Starting DD_CAL: {}\n".format("-"*60, time), f=lf, verbose=verbose)
    cal_kwargs = dict(algs['gen_cal'].items() + algs['dd_cal'].items())
    utils.log(json.dumps(cal_kwargs, indent=1) + '\n', f=lf, verbose=verbose)
    p = Dict2Obj(**cal_kwargs)

    # make a proper CASA spectral cube
    imname = '{}.{}'.format(datafile, p.model_ext)
    utils.log("...making a dummmy CASA spectral cube {}".format(imname), f=lf, verbose=verbose)
    cmd = casa + ['-c', "tclean(vis='{}', imagename='{}', niter=0, cell='{}arcsec', " \
                            "imsize={}, spw='', specmode='cube', start=0, width=300, stokes='{}')" \
                            "".format(datafile, imname, p.pxsize, p.imsize, p.stokes)]
    ecode = subprocess.check_call(cmd)

    # export to fits
    utils.log("...exporting to fits", f=lf, verbose=verbose)
    cmd = casa + ['-c', "exportfits('{}', '{}', overwrite={}, stokeslast=False)".format(imname+'.image', imname+".image.fits", overwrite)]
    ecode = subprocess.check_call(cmd)

    # erase all the original CASA files
    files = [f for f in glob.glob("{}*".format(imname)) if '.fits' not in f]
    for f in files:
        try:
            shutil.rmtree(f)
        except:
            os.remove(f)

    # make a spectral model of the imaged sources
    utils.log("...making spectral model of {}".format(p.inp_images), f=lf, verbose=verbose)
    inp_images = sorted(glob.glob(p.inp_images))
    assert len(inp_images) > 0, "nothing found under glob({})".format(p.inp_images)
    cmd = ["make_model_cube.py"] + inp_images \
        + ["--cubefile", imname + '.image.fits', '--sourcefile', p.sourcefile,
           '--outfname', imname + '.image.fits', "--makeplots", "--rb_Npix", p.rb_Npix,
           "--gp_ls", p.gp_ls, "--gp_nl", p.gp_nl,
           "--taper_alpha", p.taper_alpha, '--search_frac', p.search_frac]
    if overwrite:
        cmd += ['--overwrite']
    if p.fit_pl:
        cmd += ['--fit_pl']
    if p.fit_gp:
        cmd += ['--fit_gp']
    cmd += ['--exclude_sources'] + p.exclude_sources
    cmd = map(str, cmd)

    ecode = subprocess.check_call(cmd)

    # importfits
    utils.log("...importing from FITS", f=lf, verbose=verbose)
    cmd = casa +  ['-c', "importfits('{}', '{}', overwrite={})".format(imname+'.image.fits', imname+'.image', overwrite)]
    ecode = subprocess.check_call(cmd)
 
    # make a new flux model
    utils.log("...making new flux model for peeled visibilities, drawing parameters from gen_model", f=lf, verbose=verbose)
    utils.log(json.dumps(algs['gen_model'], indent=1) + '\n', f=lf, verbose=verbose)

    # First generate a clean_sources that excludes certain sources
    secondary_sourcefile = "{}_secondary.tab".format(os.path.splitext(p.sourcefile)[0])
    with open(secondary_sourcefile, "w") as f:
        f1 = open(p.sourcefile).readlines()
        f.write(''.join([l for i, l in enumerate(f1) if i-1 not in p.exclude_sources]))

    # Generate a New Model
    cal_kwargs['sourcefile'] = secondary_sourcefile
    model = gen_model(**dict(cal_kwargs + global_vars(varlist).items()))

    # uvsub model from corrected data
    utils.log("...uvsub CORRECTED - MODEL --> CORRECTED", f=lf, verbose=verbose)
    cmd = casa + ["-c", "uvsub('{}')".format(datafile)]
    ecode = subprocess.check_call(cmd)

    # split corrected
    split_datafile = "{}{}{}".format(os.path.splitext(datafile)[0], p.file_ext, os.path.splitext(datafile)[1])
    utils.log("...split CORRECTED to {}".format(split_datafile))
    cmd = casa + ["-c", "split('{}', '{}', datacolumn='corrected')".format(datafile, split_datafile)]
    ecode = subprocess.check_call(cmd)

    # Recalibrate
    utils.log("...recalibrating with peeled visibilities", f=lf, verbose=verbose)
    gaintables = calibrate(**dict(cal_kwargs.items() + global_vars(varlist).items()))

    # apply gaintables to datafile
    utils.log("...applying all gaintables \n\t{}\nto {}".format('\n\t'.join(gaintables), datafile), f=lf, verbose=verbose)
    cmd = casa + ['-c', 'sky_cal.py', '--msin', datafile, '--gaintables'] + gaintables
    ecode = subprocess.check_call(cmd)

    # Perform Imaging
    image(**dict(cal_kwargs.items() + global_vars(varlist).items()))

    # end block
    time2 = datetime.utcnow()
    utils.log("...finished DD_CAL: {:d} sec elapsed".format(utils.get_elapsed_time(time, time2)), f=lf, verbose=verbose)

