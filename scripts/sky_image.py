#!/usr/bin/env python2.7
"""
sky_image.py
-------------
Visibility imaging with CASA 5.1.1

run this script as:
casa -c sky_image.py <args>

Nick Kern
nkern@berkeley.edu
Sept. 2018
"""
import sys
import os
import numpy as np
import argparse
import subprocess
import shutil
import glob
from multiprocessing import Pool
import traceback
import copy

## Set Arguments
# Required Arguments
a = argparse.ArgumentParser(description="Run with casa as: casa -c sky_image.py <args>")
a.add_argument('--script', '-c', type=str, help='name of this script', required=True)
a.add_argument('--msin', default=None, type=str, help='path to a CASA measurement set. if fed a .uvfits, will convert to ms', required=True)
# IO Arguments
a.add_argument("--cleanspace", default=True, type=bool, help="Clean directory of image stem namespace before proceeding.")
a.add_argument('--source', default=None, type=str, help='Name of the main source in the field.')
a.add_argument('--out_dir', default=None, type=str, help='output directory')
a.add_argument("--silence", default=False, action='store_true', help="turn off output to stdout")
a.add_argument('--source_ext', default=None, type=str, help="Extension to default source name in output image files")
a.add_argument("--im_stem", default=None, type=str, help="Imagename stem for output images. Default is basename of input MS.")
a.add_argument("--logfile", default="output.log", type=str, help="Logging file.")
# Imaging Arguments
a.add_argument("--model", default=None, type=str, help="Path to model image or component list with *.cl suffix to insert into MODEL column.")
a.add_argument('--image_mfs', default=False, action='store_true', help="make an MFS image across the selected band")
a.add_argument('--niter', default=[0], type=int, nargs='*', help='Total number of clean iterations. Can be a list of niter for each mask provided.')
a.add_argument("--cycleniter", default=[1000], type=int, nargs='*', help="Maximum number of minor-cycle iterations before triggering a major cycle.")
a.add_argument('--pxsize', default=300, type=int, help='pixel (cell) scale in arcseconds')
a.add_argument('--imsize', default=500, type=int, help='number of pixels along a side of the output square image.')
a.add_argument('--spw', default="", type=str, help="Imaging spectral window selection.")
a.add_argument('--uvrange', default="", type=str, help="CASA uvrange string to set in imaging.")
a.add_argument('--timerange', default=[""], type=str, nargs='*', help="Imaging timerange(s)")
a.add_argument('--spec_cube', default=False, action='store_true', help="image spectral cube as well as MFS.")
a.add_argument('--spec_dchan', default=40, type=int, help="number of channel averaging for a single image in the spectral cube.")
a.add_argument('--spec_start', default=100, type=int, help='starting channel for spectral cube')
a.add_argument('--spec_end', default=924, type=int, help='ending channel for spectral cube')
a.add_argument("--stokes", default='I', type=str, help="Polarizations to image. Cannot mix Stokes and Dipole pols. Ex. 'IQUV' or 'XXYY'. Default is 'I'")
a.add_argument("--mask", default=[''], type=str, nargs='*', help="CASA region string (or lists of them) to use as mask in CLEANing. Ex: 'circle[[1h55m0s,-30d40m0s],10deg]'")
a.add_argument("--weighting", default='briggs', type=str, help="Visibility weighting when imaging.")
a.add_argument("--robust", default=0, type=float, help="Robust parameter when briggs weighting.")
a.add_argument("--gain", default=0.1, type=float, help="Gain parameter in CLEAN task: sets ratio of image peaks to insert into CLEAN model at each minor cycle.")
a.add_argument('--ex_ants', default=None, type=str, help='bad antennas to flag')
a.add_argument('--rflag', default=False, action='store_true', help='run flagdata(mode=rflag)')
a.add_argument('--unflag', default=False, action='store_true', help='start by unflagging data')
a.add_argument('--flag_autos', default=True, type=bool, help="flag autocorrelations in data.")
a.add_argument("--export_fits", default=True, type=bool, help="Export all output CASA images to FITS.")
a.add_argument("--threshold", default=['0.0mJy'], type=str, nargs='*', help="Global CLEAN stopping threshold. One for each mask provided.")
a.add_argument("--minpsffraction", default=0.1, type=float, help="PSF fraction that marks max depth for cleaning per minor cycle. A higher value triggers a major cycle sooner.")
a.add_argument("--deconvolver", default="clark", type=str, help="Algorithm for deconvolution.")
a.add_argument("--pblimit", default=-1, type=float, help="Ratio of peak primary beam response, outside of which image is masked. Default is no masking.")
a.add_argument("--multiprocess", default=False, action='store_true', help="Try to multiprocess certain parts of the pipeline. Currently mp enabled portions are: spectral cube imaging.")
a.add_argument("--Nproc", default=4, type=int, help="Number of processing to spawn in pooling.")
a.add_argument("--savemodel", default=False, action='store_true', help="When CLEANing, store FT of model components in MODEL column of MS.")
a.add_argument("--uvsub", default=False, action='store_true', help="Before imaging, subtract MODEL column from CORRECTED_DATA column if it exists (or DATA column otherwise).")
a.add_argument("--gridder", default='wproject', type=str, help="Gridding algorithm in tclean. Options=['standard', 'wproject', 'widefield']. Note that gridder='wproject' and wprojplanes=1 is equivalent to gridder='standard'.")
a.add_argument("--wprojplanes", default=1, type=int, help="Number of W-values to use for computing W-projection kernel. Default=1 is standard non-Wprojection gridding.")
# Plotting Arguments
a.add_argument("--plot_uvdist", default=False, action='store_true', help='make a uvdist plot')

# Define log function
def log(msg, f=None, tb=None, type=0, verbose=True):
    """
    Add a message to the log.
    
    Parameters
    ----------
    msg : str
        Message string to print.

    f : file descriptor
        file descriptor to write message to.

    tb : traceback tuple, optional
        Output of sys.exc_info()
    """
    if type == 1:
        msg = "\n{}\n{}".format(msg, '-'*40)

    # catch for traceback if provided
    if tb is not None:
        msg += "\n{}".format('\n'.join(traceback.format_exception(*tb)))

    # print
    if verbose:
        print(msg)

    # write
    if f is not None:
        f.write(msg)
        f.flush()

# Define CLEAN Function
def image_mfs(d):
    """ MFS Imaging Function via CLEAN task.
        d should be a dictionary holding all required parameters
        from the argparser
    """
    try:
        for i, (n, cn, m, t) in enumerate(zip(d['niter'], d['cycleniter'], d['mask'], d['threshold'])):
            log("...cleaning {} for {} iters with mask '{}'".format(d['msin'], n, m))
            # erase mask if it exists
            old_m = "{}.mask".format(d['im_stem'])
            if os.path.exists(old_m) and m != old_m:
                shutil.move(old_m, old_m+'_{}'.format(i))
            tclean(vis=d['msin'], imagename=d['im_stem'], spw=d['spw'], niter=n, cycleniter=cn, weighting=d['weighting'], robust=d['robust'], imsize=d['imsize'],
                  cell='{}arcsec'.format(d['pxsize']), specmode='mfs', timerange=d['timerange'], uvrange=d['uvrange'], stokes=d['stokes'],
                  mask=m, deconvolver=d['deconvolver'], threshold=t, savemodel=d['savemodel'], gain=d['gain'],
                  pblimit=d['pblimit'], minpsffraction=d['minpsffraction'], gridder=d['gridder'], wprojplanes=d['wprojplanes'])
        if i > 0:
            shutil.move(old_m, old_m + '_{}'.format(i+1))
        log("...saving {}".format('{}.image'.format(d['im_stem'])))
        if d['export_fits']:
            exportfits(imagename='{}.image'.format(d['im_stem']), fitsimage='{}.image.fits'.format(d['im_stem']), stokeslast=False)
            log("...saving {}".format('{}.image.fits'.format(d['im_stem'])))

    except:
        log("CLEANing threw an error:\n", f=lf, tb=sys.exc_info())

if __name__ == "__main__":
    # parse args
    args = a.parse_args()

    # parse special arguments
    if args.savemodel:
        args.savemodel = 'modelcolumn'
    else:
        args.savemodel = 'none'

    # get ms
    msin = args.msin

    # get vars
    if args.source_ext is None:
        args.source_ext = ''
    verbose = args.silence is False

    # open logfile
    lf = open(args.logfile, 'w')

    # setup multiprocessing if requested
    if args.multiprocess:
        pool = Pool(args.Nproc)
        M = pool.map
    else:
        M = map

    # Insert a model if desired (only relevant if asking for uvsub as well)
    if args.model is not None:
        if os.path.splitext(args.model)[1] == '.cl':
            log("...inserting component list {} as MODEL".format(args.model), type=1)
            ft(msin, complist=args.model, usescratch=True)
        else:
            log("...inserting image {} as MODEL".format(args.model), type=1)
            ft(msin, model=args.model, usescratch=True)

    # get phase center
    if args.source is not None:
        ra, dec = np.loadtxt('{}.loc'.format(args.source), dtype=str)
        ra, dec = ra.split(':'), dec.split(':')
        fixdir = 'J2000 {}h{}m{}s {}d{}m{}s'.format(*(ra+dec))
    else:
        args.source = ''
        fixdir = None

    # get paths
    base_ms = os.path.basename(msin)
    if args.out_dir is None:
        out_dir = os.path.dirname(msin)
    else:
        out_dir = args.out_dir

    # check for uvfits
    if base_ms.split('.')[-1] == 'uvfits':
        log("...converting uvfits to ms", type=1, verbose=verbose)
        uvfits = msin
        msin = os.path.join(out_dir, '.'.join(base_ms.split('.')[:-1] + ['ms']))
        base_ms = os.path.basename(msin)
        msfiles = glob.glob("{}*".format(msin))
        if len(msfiles) != 0:
            for i, msf in enumerate(msfiles):
                try:
                    os.remove(msf)
                except OSError:
                    shutil.rmtree(msf)
        importuvfits(uvfits, msin)
        log("creating {}".format(msin))

    # rephase to source
    if fixdir is not None:
        log("...fix vis to {} at {}".format(args.source, fixdir), type=1, verbose=verbose)
        fixvis(msin, msin, phasecenter=fixdir)

    # unflag
    if args.unflag is True:
        log("...unflagging", type=1)
        flagdata(msin, mode='unflag')

    # flag autocorrs
    if args.flag_autos:
        log("...flagging autocorrs", type=1)
        flagdata(msin, autocorr=True)

    # flag bad ants
    if args.ex_ants is not None:
        args.ex_ants = ','.join(map(lambda x: "HH"+x, args.ex_ants.split(',')))
        log("...flagging bad ants: {}".format(args.ex_ants), type=1)
        flagdata(msin, mode='manual', antenna=args.ex_ants)

    # rflag
    if args.rflag is True:
        log("...rfi flagging", type=1)
        flagdata(msin, mode='rflag')

    # get image stem
    if args.im_stem is None:
        im_stem = os.path.join(out_dir, '.'.join([base_ms, args.source + args.source_ext]))
    else:
        im_stem = args.im_stem
  
    if args.cleanspace:
        # remove paths
        log("...cleaning namespace {}".format(im_stem+'*'), type=1)
        source_files = glob.glob(im_stem+'*')
        if len(source_files) > 0:
            for sf in source_files:
                if os.path.exists(sf):
                    try:
                        shutil.rmtree(sf)
                    except OSError:
                        os.remove(sf)

    # uvsub if desired
    if args.uvsub:
        log("...performing uvsub of CORRECTED -= MODEL on {}".format(msin), type=1)
        uvsub(msin)

    # create mfs image
    if args.image_mfs:
        log("...running MFS", type=1)
        # resolve multi-mask runs
        Nniter, Nmask, Ncniter, Nthresh = len(args.niter), len(args.mask), len(args.cycleniter), len(args.threshold)
        if Nniter > 1 or Nmask > 1 or Ncniter > 1 or Nthresh > 1:
            Nmax = np.max([Nniter, Nmask, Ncniter, Nthresh])
            if Nniter == 1:
                args.niter = args.niter * Nmax
            if Nmask == 1:
                args.mask = args.mask * Nmax
            if Ncniter == 1:
                args.cycleniter = args.cycleniter * Nmax
            if Nthresh == 1:
                args.threshold = args.threshold * Nmax
        assert len(args.niter) == len(args.mask) == len(args.cycleniter) == len(args.threshold), "len(niter) must == len(mask) must == len(cycleniter) must == len(threshold)"

        # setup parameter dictionaries
        param_dicts = [copy.deepcopy(vars(args)) for i in args.timerange]
        for i, tr in enumerate(args.timerange):
            if i == 0:
                tr_im_stem = im_stem
            else:
                tr_im_stem = im_stem+'._tr{}'.format(i)
            param_dicts[i]['im_stem'] = tr_im_stem
            param_dicts[i]['timerange'] = tr

        # loop over time ranges and call imaging function
        M(image_mfs, param_dicts)

    # create spectrum
    if args.spec_cube:
        log("...running MFS spectral cube clean", type=1)
        assert len(args.niter) == len(args.mask), "len(niter) must equal len(mask)"

        # setup parameter dictionaries
        spec_windows = np.arange(args.spec_start, args.spec_end, args.spec_dchan)
        # loop over list of time-ranges
        for i, tr in enumerate(args.timerange):
            param_dicts = [copy.deepcopy(vars(args)) for j in spec_windows]

            # loop over spectral windows
            for j, chan in enumerate(spec_windows):
                spec_im_stem = '{}.spec{:04d}'.format(im_stem, chan)
                if i != 0:
                    spec_im_stem += '._tr{}'.format(j)

                param_dicts[j]['im_stem'] = spec_im_stem
                param_dicts[j]['timerange'] = tr
                param_dicts[j]['spw'] = '0:{}~{}'.format(chan, chan+args.spec_dchan-1)

            # loop over spectral windows
            M(image_mfs, param_dicts)

    # make uvdist plot
    if args.plot_uvdist:
        log("...plotting uvdistance", type=1)
        # load visibility amplitudes
        ms.open(msin)
        data = ms.getdata(["amplitude", "antenna1", "antenna2", "uvdist", "axis_info", "flag"], ifraxis=True)
        amps = data['amplitude']
        flags = data['flag']
        uvdist = data['uvdist']
        freqs = data['axis_info']['freq_axis']['chan_freq']
        ms.close()
        # get rid of autos
        select = []
        for i, a1 in enumerate(data['antenna1']):
            if a1 != data['antenna2'][i]:
                select.append(i)
        amps = amps[:, :, select, :].squeeze()
        uvdist = uvdist[select, :]
        flags = flags[:, :, select, :].squeeze()
        # omit flagged data
        amps[flags] *= np.nan
        # average across time
        amps = np.nanmean(amps, axis=2)
        uvdist = np.nanmean(uvdist, axis=1)
        # average into channel bins
        freq_bins = np.median(np.split(np.linspace(100., 200., 1024, endpoint=True)[100:924], 4), axis=1)
        amps = np.nanmedian(np.split(amps[100:924, :], 4), axis=1)
        # plot
        import matplotlib.pyplot as plt
        def plot_uvdist(amp, ext):
            fig, ax = plt.subplots(1, 1, figsize=(10,7))
            ax.grid(True)
            p = ax.plot(uvdist, amp.T, marker='o', markersize=6, alpha=0.8, ls='')
            ax.set_xlabel("baseline length (meters)", fontsize=16)
            ax.set_ylabel("amplitude (arbitrary units)", fontsize=16)
            ax.set_title("{} for {}".format(ext, base_ms))
            ax.legend(p, map(lambda x: '{:0.0f} MHz'.format(x), freq_bins))
            file_fname = os.path.join(out_dir, "{}.{}.png".format(base_ms, ext))
            log("...saving {}".format(file_fname))
            fig.savefig(file_fname, bbox_inches='tight', pad=0.05)
            plt.close()

        plot_uvdist(amps, "uvdist")

    # close processes
    if args.multiprocess:
        pool.close()

