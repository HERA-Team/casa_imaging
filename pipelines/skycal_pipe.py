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
data_file = os.path.join(params['data_root'], params['data_file'])

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
# Search for a Source in the Data and Prepare for MS Conversion
#-------------------------------------------------------------------------------
if params['prep_data']:
    # start block
    time = datetime.utcnow()
    utils.log("\n{}\n...Starting PREP_DATA: {}\n".format("-"*60, time), 
                 f=lf, verbose=verbose)
    utils.log(json.dumps(algs['prep_data'], indent=1) + '\n', f=lf, verbose=verbose)
    p = Dict2Obj(**algs['prep_data'])

    # Check if data_file is already MS
    import source2file
    if os.path.splitext(data_file)[1] == '.ms':
        # get transit times
        (lst, transit_jd, utc_range, utc_center, source_files,
         source_utc_range) = source2file.source2file(point_ra, lon=longitude, lat=latitude,
                                                     duration=p.duration, start_jd=p.start_jd, get_filetimes=False,
                                                     verbose=verbose)
        utils.log("...the file {} is already a CASA MS, skipping rest of PREP_DATA".format(data_file), f=lf, verbose=verbose)
        timerange = utc_range
        datafile = data_file

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
            datafiles = [df for df in glob.glob(data_file) if pol in df]
            assert len(datafiles) > 0, "Searching for {} with pol {} but found no files...".format(data_file, pol)

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
                        "data template {}".format(data_file), f=lf, verbose=verbose)
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

    # overwrite gen_model pbcorr transit jd
    algs['gen_model']['pbcorr_params']['time'] = transit_jd

    # end block
    time2 = datetime.utcnow()
    start = time.day*24*3600 + time.hour*3600 + time.minute*60 + time.second
    end = time2.day*24*3600 + time2.hour*3600 + time2.minute*60 + time2.second
    utils.log("...finished PREP_DATA: {:d} sec elapsed".format(end-start), f=lf, verbose=verbose)

#-------------------------------------------------------------------------------
# Generate Flux Model
#-------------------------------------------------------------------------------
if params['gen_model']:
    # start block
    time = datetime.utcnow()
    utils.log("\n{}\n...Starting GEN_MODEL: {}\n".format("-"*60, time), 
                 f=lf, verbose=verbose)
    utils.log(json.dumps(algs['gen_model'], indent=1) + '\n', f=lf, verbose=verbose)
    p = Dict2Obj(**algs['gen_model'])
    c = Dict2Obj(**p.cl_params)

    # compile command
    cmd = casa + ["-c", "complist_gleam.py"]
    cmd += ['--point_ra', point_ra, '--point_dec', latitude, '--outdir', out_dir, 
            '--gleamfile', c.gleamfile, '--radius', c.radius, '--min_flux', c.min_flux,
            '--freqs', c.freqs, '--cell', c.cell, '--imsize', c.imsize]
    if c.image:
        cmd += ['--image']
    if c.use_peak:
        cmd += ['--use_peak']
    if overwrite:
        cmd += ['--overwrite']
    cmd = map(str, cmd)

    # generate component list and / or image cube flux model
    ecode = subprocess.check_call(cmd)
    model = os.path.join(out_dir, "gleam.cl")
    if p.cl_params['image']:
        model = os.path.join(out_dir, "gleam.cl.image")

    # pbcorrect
    if p.pbcorr:
        assert p.cl_params['image'], "Cannot pbcorrect flux model without image == True"
        cmd = ["pbcorr.py", "--lon", str(longitude), "--lat", str(latitude), "--time",
               str(p.pbcorr_params['time']), "--pols"] \
               + [str(uvutils.polstr2num(pol)) for pol in p.pbcorr_params['pols']] \
               + ["--outdir", out_dir, "--multiply", "--beamfile", p.pbcorr_params['beamfile']]
        if overwrite:
            cmd.append("--overwrite")
        cmd.append(os.path.join(out_dir, "gleam.cl.fits"))

        # generate component list and / or image cube flux model
        ecode = subprocess.check_call(cmd)

        # importfits
        cmd = casa + ["-c", "importfits('{}}', '{}', overwrite={})".format(os.path.join(out_dir, 'gleam.cl.pbcorr.fits'), os.path.join(out_dir, 'gleam.cl.pbcorr.image'), overwrite)]
        ecode = subprocess.check_call(cmd)
        model = os.path.join(out_dir, "gleam.cl.pbcorr.image")

    # update di_cal model path
    algs['di_cal']['cal_params']['model'] = model

    # end block
    time2 = datetime.utcnow()
    start = time.day*24*3600 + time.hour*3600 + time.minute*60 + time.second
    end = time2.day*24*3600 + time2.hour*3600 + time2.minute*60 + time2.second
    utils.log("...finished GEN_MODEL: {:d} sec elapsed".format(end-start), f=lf, verbose=verbose)


#-------------------------------------------------------------------------------
# Direction Independent Calibration
#-------------------------------------------------------------------------------
if params['di_cal']:
    # start block
    time = datetime.utcnow()
    utils.log("\n{}\n...Starting DI_CAL: {}\n".format("-"*60, time), f=lf, verbose=verbose)
    utils.log(json.dumps(algs['di_cal'], indent=1) + '\n', f=lf, verbose=verbose)
    p = Dict2Obj(**dict(algs['gen_cal'].items() + algs['di_cal'].items()))

    # If source.loc file doesn't exist, write it
    if not os.path.exists("{}.loc".format(p.source)):
        ra = p.source_ra / 15.0
        ra_h = int(np.floor(ra))
        ra_m = int(np.floor((ra - ra_h) * 60))
        ra_s = int(np.around(((ra - ra_h) * 60 - ra_m) * 60))
        dec_d = int(np.floor(np.abs(p.source_dec)) * p.source_dec / np.abs(p.source_dec))
        dec_m = int(np.floor(np.abs(p.source_dec - dec_d) * 60.))
        dec_s = int(np.abs(p.source_dec - dec_d) * 3600 - dec_m * 60)
        direction = "{:02d}:{:02d}:{:02.0f}\t{:03d}:{:02d}:{:02.0f}:".format(ra_h, ra_m, ra_s, dec_d, dec_m, dec_s)
        with open('{}.loc'.format(p.source), 'w') as f:
            f.write(direction)

    # Perform Calibration
    c = Dict2Obj(**p.cal_params)
    cmd = casa + ["-c", "sky_cal.py"]
    cmd += ["--msin", datafile, "--source", p.source, "--out_dir", out_dir, "--model", c.model,
            "--refant", c.refant, "--gain_spw", c.gain_spw, "--uvrange", c.uvrange, "--timerange",
            c.timerange, "--ex_ants", p.ex_ants, "--gain_ext", c.gain_ext, '--bp_spw', c.bp_spw]
    if isinstance(c.gaintables, list):
        gaintables = c.gaintables
    else:
        if c.gaintables in ['', None, 'None', 'none']:
            gaintables = []
        else:
            gaintables = [c.gaintables]
    if len(gaintables) > 0:
        cmd += ['--gaintables'] +  gaintables
    if c.rflag:
        cmd += ["--rflag"]
    if c.KGcal:
        cmd += ["--KGcal", "--KGsnr", c.KGsnr]
    if c.Acal:
        cmd += ["--Acal", "--Asnr", c.Asnr]
    if c.BPcal:
        cmd += ["--BPcal", "--BPsnr", c.BPsnr, "--bp_spw", c.bp_spw]
    if c.BPsolnorm:
        cmd += ['--BPsolnorm']
    if c.split_cal:
        cmd += ["--split_cal", "--cal_ext", c.cal_ext]
    if c.split_model:
        cmd += ["--split_model"]
    cmd = [' '.join(_cmd) if type(_cmd) == list else str(_cmd) for _cmd in cmd]

    utils.log("...starting calibration", f=lf, verbose=verbose)
    ecode = subprocess.check_call(cmd)

    # Gather gaintables
    gext = ''
    if c.gain_ext not in ['', None]:
        gext = '.{}'.format(c.gain_ext)
    gts = sorted(glob.glob("{}{}.?.cal".format(datafile, gext)) + glob.glob("{}{}.????.cal".format(datafile, gext)))

    # export to calfits if desired
    if p.export_gains:
        utils.log("...exporting\n{}\n to calfits and combining into a single cal table".format('\n'.join(gts)), f=lf, verbose=verbose)
        # do file checks
        mirvis = os.path.splitext(data_file)[0]
        gtsnpz = ["{}.npz".format(gt) for gt in gts]
        if not os.path.exists(mirvis):
            utils.log("...{} doesn't exist: cannot export gains to calfits".format(mirvis), f=lf, verbose=verbose)

        elif len(gts) == 0:
            utils.log("...no gaintables found, cannot export gains to calfits", f=lf, verbose=verbose)

        elif not np.all([os.path.exists(gtnpz) for gtnpz in gtsnpz]):
            utils.log("...couldn't find a .npz file for all input .cal tables, can't export to calfits", f=lf, verbose=verbose)

        else:
            calfits_fname = "{}.{}{}.calfits".format(mirvis, p.source, c.gain_ext)
            cmd = ['skynpz2calfits.py', "--fname", calfits_fname, "--uv_file", mirvis, '--out_dir', out_dir]
            if overwrite:
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
            if p.export_params['smooth']:
                cmd += ["--bp_gp_smooth", "--bp_gp_max_dly", p.export_params['gp_max_dly']]
            if p.export_params['medfilt']:
                cmd += ["--bp_medfilt", "--medfilt_kernel", p.export_params['kernel']]
            if p.export_params['bp_broad_flags']:
                cmd += ["--bp_broad_flags", "--bp_flag_frac", p.export_params['bp_flag_frac']]
            if not verbose:
                cmd += ['--silence']

            cmd = map(str, cmd)
            ecode = subprocess.check_call(cmd)

            # convert calfits back to a single Btotal.cal table
            if np.any(matchB):
                # convert to cal
                bfile = gts[matchB.index(True)]
                btot_file = os.path.join(out_dir, "{}{}.Btotal.cal".format(datafile, gext))
                cmd = casa + ["-c", "calfits_to_Bcal.py", "--cfits", calfits_fname, "--inp_cfile", bfile,"--out_cfile", btot_file]
                if overwrite:
                    cmd += ["--overwrite"]
                ecode = subprocess.check_call(cmd)
                # replace gaintables with Btotal.cal
                gts = [btot_file]

    # append to gaintables
    try:
        gaintables += gts
    except NameError:
        gaintables = gts

    # Perform MFS imaging
    if p.image_mfs:
        # Image Corrected Data
        utils.log("...starting MFS image of CORRECTED data", f=lf, verbose=verbose)
        c = Dict2Obj(**p.img_params)
        cmd = casa + ["-c", "sky_image.py"]
        cmd += ["--source", p.source, "--out_dir", out_dir,
                "--pxsize", c.pxsize, "--imsize", c.imsize,
                "--spw", c.spw, "--uvrange", c.uvrange, "--timerange", c.timerange,
                "--stokes", c.stokes, "--weighting", c.weighting, "--robust", c.robust,
                "--pblimit", c.pblimit, "--deconvolver", c.deconvolver, "--niter",
                c.niter, '--cycleniter', c.cycleniter, '--threshold', c.threshold,
                '--mask', c.mask]
        cmd = [map(str, _cmd) if type(_cmd) == list else str(_cmd) for _cmd in cmd]
        cmd = reduce(operator.add, [i if type(i) == list else [i] for i in cmd])
        icmd = cmd + ['--image_mfs', '--msin', datafile, "--source_ext", c.source_ext] 
        ecode = subprocess.check_call(icmd)

        # Image the split MODEL
        if c.image_mdl:
            utils.log("...starting MFS image of MODEL data", f=lf, verbose=verbose)
            mfile = "{}.model".format(datafile)
            if not os.path.exists(mfile):
                utils.log("Didn't split model from datafile, which is required to image the model", f=lf, verbose=verbose)
            else:
                icmd = cmd + ['--image_mfs', '--msin', mfile, '--source_ext', c.source_ext]
                ecode = subprocess.check_call(icmd)

        # Image the CORRECTED - MODEL residual
        if c.image_res:
            utils.log("...starting MFS image of CORRECTED - MODEL data", f=lf, verbose=verbose)
            icmd = cmd + ['--image_mfs', '--msin', datafile, '--uvsub', "--source_ext", "{}_resid".format(c.source_ext)] 
            ecode = subprocess.check_call(icmd)

            # Apply gaintables to make CORRECTED column as it was
            utils.log("...reapplying gaintables to CORRECTED data", f=lf, verbose=verbose)
            cmd2 = casa + ["-c", "sky_cal.py", "--msin", datafile, "--gaintables"] + gaintables
            ecode = subprocess.check_call(cmd2)

    if p.image_spec:
        # Perform Spectral Cube imaging
        utils.log("...starting spectral cube imaging", f=lf, verbose=verbose)
        icmd = cmd + ['--spec_cube', '--msin', datafile, "--source_ext", c.source_ext,
                      '--spec_start', str(c.spec_start), '--spec_end', str(c.spec_end),
                      '--spec_dchan', str(c.spec_dchan)] 
        ecode = subprocess.check_call(icmd)

    # end block
    time2 = datetime.utcnow()
    start = time.day*24*3600 + time.hour*3600 + time.minute*60 + time.second
    end = time2.day*24*3600 + time2.hour*3600 + time2.minute*60 + time2.second
    utils.log("...finished DI_CAL: {:d} sec elapsed".format(end-start), f=lf, verbose=verbose)

#-------------------------------------------------------------------------------
# Direction Dependent Calibration
#-------------------------------------------------------------------------------
if params['dd_cal']:
    # start block
    time = datetime.utcnow()
    utils.log("\n{}\n...Starting DI_CAL: {}\n".format("-"*60, time), f=lf, verbose=verbose)
    utils.log(json.dumps(algs['dd_cal'], indent=1) + '\n', f=lf, verbose=verbose)
    p = Dict2Obj(**dict(algs['gen_cal'].items() + algs['dd_cal'].items()))

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
    utils.log("...making new flux model for peeled visibilities, drawing parameters from gen_model.cl_params", f=lf, verbose=verbose)
    utils.log(json.dumps(algs['gen_model']['cl_params'], indent=1) + '\n', f=lf, verbose=verbose)
    gm = Dict2Obj(**algs['gen_model'])
    cl = Dict2Obj(**algs['gen_model']['cl_params'])
    pb = Dict2Obj(**algs['gen_model']['pbcorr_params'])

    # First generate a clean_sources that excludes certain sources
    secondary_sourcefile = "{}_secondary.tab".format(os.path.splitext(p.sourcefile)[0])
    with open(secondary_sourcefile, "w") as f:
        f1 = open(p.sourcefile).readlines()
        f.write(''.join([l for i, l in enumerate(f1) if i-1 not in p.exclude_sources]))

    # compile command
    cmd = casa + ['-c', 'complist_gleam.py', '--point_ra', str(point_ra), '--point_dec', str(latitude),
                  '--outdir', out_dir, '--gleamfile', cl.gleamfile, "--radius", cl.radius, "--min_flux",
                  cl.min_flux, "--freqs", cl.freqs, "--cell", cl.cell,
                  "--imsize", cl.imsize, "--ext", p.file_ext, '--region_radius', p.region_radius,
                  '--exclude', '--regions', secondary_sourcefile]
    if overwrite:
        cmd += ['--overwrite']
    if cl.image:
        cmd += ['--image']
    if cl.use_peak:
        cmd += ['--use_peak']
    cmd = map(str, cmd)
    ecode = subprocess.check_call(cmd)
    model = "gleam{}.cl".format(p.file_ext)
    if cl.image:
        model += ".fits"

    # pbcorrect
    if gm.pbcorr:
        assert cl.image, "Cannot pbcorrect flux model without image == True"
        utils.log("...multipying primary beam into new flux model", f=lf, verbose=verbose)
        cmd = ["pbcorr.py", "--lon", str(longitude), "--lat", str(latitude), "--time",
               str(pb.time), "--pols"] \
               + [str(uvutils.polstr2num(pol)) for pol in pb.pols] \
               + ["--outdir", out_dir, "--multiply", "--beamfile", pb.beamfile]
        if overwrite:
            cmd.append("--overwrite")
        cmd += [model]

        # primary beam correct image
        ecode = subprocess.check_call(cmd)

        # importfits
        new_model = '{}.pbcorr'.format(os.path.splitext(model)[0])
        cmd = casa + ["-c", "importfits('{}.fits', '{}.image', overwrite={})".format(new_model, new_model, overwrite)]
        ecode = subprocess.check_call(cmd)
        model = "{}.image".format(new_model)

    # uvsub model from corrected data
    utils.log("...uvsub CORRECTED - MODEL --> CORRECTED", f=lf, verbose=verbose)
    cmd = casa + ["-c", "uvsub('{}')".format(datafile)]
    ecode = subprocess.check_call(cmd)

    # split corrected
    split_datafile = "{}{}{}".format(os.path.splitext(datafile)[0], p.file_ext, os.path.splitext(datafile)[1])
    utils.log("...split CORRECTED to {}".format(split_datafile))
    cmd = casa + ["-c", "split('{}', '{}', datacolumn='corrected')".format(datafile, split_datafile)]
    ecode = subprocess.check_call(cmd)

    # recalibrate
    c = Dict2Obj(**p.cal_params)
    cmd = casa + ["-c", "sky_cal.py"]
    cmd += ["--msin", split_datafile, "--source", p.source, "--out_dir", out_dir, "--model", model,
            "--refant", c.refant, "--gain_spw", c.gain_spw, "--uvrange", c.uvrange, "--timerange",
            c.timerange, "--ex_ants", p.ex_ants, "--gain_ext", c.gain_ext, '--bp_spw', c.bp_spw]

    # parse fed gaintables
    if not isinstance(c.gaintables, list):
        if c.gaintables in ['', None, 'None', 'none']:
            c.gaintables = []
        else:
            c.gaintables = [c.gaintables]
    if len(c.gaintables) > 0:
        cmd += ['--gaintables'] + c.gaintables

    # parse extra flags
    if c.rflag:
        cmd += ["--rflag"]
    if c.KGcal:
        cmd += ["--KGcal", "--KGsnr", c.KGsnr]
    if c.Acal:
        cmd += ["--Acal", "--Asnr", c.Asnr]
    if c.BPcal:
        cmd += ["--BPcal", "--BPsnr", c.BPsnr, "--bp_spw", c.bp_spw]
    if c.BPsolnorm:
        cmd += ['--BPsolnorm']
    if c.split_cal:
        cmd += ["--split_cal", "--cal_ext", c.cal_ext]
    if c.split_model:
        cmd += ["--split_model"]
    cmd = [' '.join(_cmd) if type(_cmd) == list else str(_cmd) for _cmd in cmd]

    utils.log("...recalibrating with peeled visibilities", f=lf, verbose=verbose)
    ecode = subprocess.check_call(cmd)

    # Gather gaintables
    gext = ''
    if c.gain_ext not in ['', None]:
        gext = '.{}'.format(c.gain_ext)
    gts = sorted(glob.glob("{}{}.?.cal".format(split_datafile, gext)) \
                        + glob.glob("{}{}.????.cal".format(split_datafile, gext)))

    # export to calfits if desired
    if p.export_gains:
        utils.log("...exporting\n{}\n to calfits and combining into a single cal table".format('\n'.join(gts)), f=lf, verbose=verbose)
        # do file checks
        mirvis = os.path.splitext(data_file)[0]
        gtsnpz = ["{}.npz".format(gt) for gt in gts]
        if not os.path.exists(mirvis):
            utils.log("...{} doesn't exist: cannot export gains to calfits".format(mirvis), f=lf, verbose=verbose)

        elif len(gts) == 0:
            utils.log("...no gaintables found, cannot export gains to calfits", f=lf, verbose=verbose)

        elif not np.all([os.path.exists(gtnpz) for gtnpz in gtsnpz]):
            utils.log("...couldn't find a .npz file for all input .cal tables, can't export to calfits", f=lf, verbose=verbose)

        else:
            calfits_fname = "{}.{}{}.calfits".format(mirvis, p.source, c.gain_ext)
            cmd = ['skynpz2calfits.py', "--fname", calfits_fname, "--uv_file", mirvis, '--out_dir', out_dir]
            if overwrite:
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
            if p.export_params['smooth']:
                cmd += ["--bp_gp_smooth", "--bp_gp_max_dly", p.export_params['gp_max_dly']]
            if p.export_params['medfilt']:
                cmd += ["--bp_medfilt", "--medfilt_kernel", p.export_params['kernel']]
            if p.export_params['bp_broad_flags']:
                cmd += ["--bp_broad_flags", "--bp_flag_frac", p.export_params['bp_flag_frac']]
            if not verbose:
                cmd += ['--silence']

            cmd = map(str, cmd)
            ecode = subprocess.check_call(cmd)

            # convert calfits back to a single Btotal.cal table
            if np.any(matchB):
                # convert to cal
                bfile = gts[matchB.index(True)]
                btot_file = os.path.join(out_dir, "{}{}.Btotal.cal".format(datafile, gext))
                cmd = casa + ["-c", "calfits_to_Bcal.py", "--cfits", calfits_fname, "--inp_cfile", bfile,"--out_cfile", btot_file]
                if overwrite:
                    cmd += ["--overwrite"]
                ecode = subprocess.check_call(cmd)
                # replace gaintables with Btotal.cal
                gts = [btot_file]

    try:
        gaintables += gts
    except NameError:
        gaintables = gts

    # apply gaintables to datafile
    utils.log("...applying all gaintables \n\t{}\nto {}".format('\n\t'.join(gaintables), datafile), f=lf, verbose=verbose)
    cmd = casa + ['-c', 'sky_cal.py', '--msin', datafile, '--gaintables'] + gaintables
    ecode = subprocess.check_call(cmd)

    # Perform MFS imaging
    if p.image_mfs:
        # Image Corrected Data
        utils.log("...starting MFS image of CORRECTED data", f=lf, verbose=verbose)
        c = Dict2Obj(**p.img_params)
        cmd = casa + ["-c", "sky_image.py"]
        cmd += ["--source", p.source, "--out_dir", out_dir,
                "--pxsize", c.pxsize, "--imsize", c.imsize,
                "--spw", c.spw, "--uvrange", c.uvrange, "--timerange", c.timerange,
                "--stokes", c.stokes, "--weighting", c.weighting, "--robust", c.robust,
                "--pblimit", c.pblimit, "--deconvolver", c.deconvolver, "--niter",
                c.niter, '--cycleniter', c.cycleniter, '--threshold', c.threshold,
                '--mask', c.mask]
        cmd = [map(str, _cmd) if type(_cmd) == list else str(_cmd) for _cmd in cmd]
        cmd = reduce(operator.add, [i if type(i) == list else [i] for i in cmd])
        icmd = cmd + ['--image_mfs', '--msin', datafile, "--source_ext", c.source_ext] 

        ecode = subprocess.check_call(icmd)

        # Image the split MODEL
        if c.image_mdl:
            utils.log("...starting MFS image of MODEL data", f=lf, verbose=verbose)
            mfile = "{}.model".format(datafile)
            if not os.path.exists(mfile):
                utils.log("Didn't split model from datafile, which is required to image the model", f=lf, verbose=verbose)
            else:
                icmd = cmd + ['--image_mfs', '--msin', mfile, '--source_ext', c.source_ext]
                ecode = subprocess.check_call(icmd)

        # Image the CORRECTED - MODEL residual
        if c.image_res:
            utils.log("...starting MFS image of CORRECTED - MODEL data", f=lf, verbose=verbose)
            icmd = cmd + ['--image_mfs', '--msin', datafile, '--uvsub', "--source_ext", "{}_resid".format(c.source_ext)] 
            ecode = subprocess.check_call(icmd)

            # Apply gaintables to make CORRECTED column as it was
            utils.log("...reapplying gaintables to CORRECTED data", f=lf, verbose=verbose)
            cmd2 = casa + ["-c", "sky_cal.py", "--msin", datafile, "--gaintables"] + gaintables
            ecode = subprocess.check_call(cmd2)

    if p.image_spec:
        # Perform Spectral Cube imaging
        utils.log("...starting spectral cube imaging", f=lf, verbose=verbose)
        icmd = cmd + ['--spec_cube', '--msin', datafile, "--source_ext", c.source_ext,
                      '--spec_start', str(c.spec_start), '--spec_end', str(c.spec_end),
                      '--spec_dchan', str(c.spec_dchan)] 
        ecode = subprocess.check_call(icmd)

    # end block
    time2 = datetime.utcnow()
    start = time.day*24*3600 + time.hour*3600 + time.minute*60 + time.second
    end = time2.day*24*3600 + time2.hour*3600 + time2.minute*60 + time2.second
    utils.log("...finished DD_CAL: {:d} sec elapsed".format(end-start), f=lf, verbose=verbose)

