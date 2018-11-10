#!/usr/bin/env python2.7
"""
sky_cal.py
-------------
Sky-based calibration with CASA 5.1.1

run this script as:
casa -c sky_cal.py <args>

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
import re

## Set Arguments
# Required Arguments
a = argparse.ArgumentParser(description="Run with casa as: casa -c sky_image.py <args>")
a.add_argument('--script', '-c', type=str, help='name of this script', required=True)
a.add_argument('--msin', default=None, type=str, help='path to a CASA measurement set. if fed a .uvfits, will convert to ms', required=True)
# IO Arguments
a.add_argument('--source', default=None, type=str, help='Name of the main source in the field.')
a.add_argument('--out_dir', default=None, type=str, help='output directory')
a.add_argument("--silence", default=False, action='store_true', help="turn off output to stdout")
a.add_argument("--gain_ext", default='', type=str, help="Extension for output gain tables, after the msin stem but before the *.cal suffix.")
# Calibration Arguments
a.add_argument("--model", default=None, type=str, help="Path to model image or component list with *.cl suffix to insert into MODEL column.")
a.add_argument('--refant', default=None, type=str, help='Reference antenna, or comma-delimited list of ref ants for backup.')
a.add_argument('--ex_ants', default=None, type=str, help='bad antennas to flag')
a.add_argument('--rflag', default=False, action='store_true', help='run flagdata(mode=rflag)')
a.add_argument('--unflag', default=False, action='store_true', help='start by unflagging data')
a.add_argument('--KGcal', default=False, action='store_true', help='perform K (dly) & G (phs) calibration')
a.add_argument("--KGsnr", default=2.0, type=float, help="KG calibration Signal-to-Noise cut")
a.add_argument('--Acal', default=False, action='store_true', help='perform G (amp) calibration')
a.add_argument("--Asnr", default=2.0, type=float, help="G-amplitude calibration Signal-to-Noise cut")
a.add_argument('--BPcal', default=False, action='store_true', help='perform BandPass calibration (phs & amp)')
a.add_argument("--BPsnr", default=2.0, type=float, help="bandpass calibration Signal-to-Noise cut")
a.add_argument("--BPsolnorm", default=False, action='store_true', help="Normalize freq average of bandpass solution amplitude to 1.0.")
a.add_argument('--uvrange', default="", type=str, help="CASA uvrange string to set in calibration.")
a.add_argument('--timerange', default=[""], type=str, nargs='*', help="Calibration timerange(s)")
a.add_argument('--bpoly', default=False, action='store_true', help="use BPOLY mode in bandpass")
a.add_argument('--degamp', default=4, type=int, help="amplitude polynomial degree for BPOLY")
a.add_argument('--degphase', default=1, type=int, help="phase polynomial degree for BPOLY")
a.add_argument('--gain_spw', default='', type=str, help="Spectral window selection for gaincal routine.")
a.add_argument("--bp_spw", default='', type=str, help="Spectral window selection for bandpass routine.")
a.add_argument('--flag_autos', default=False, action='store_true', help="flag autocorrelations in data.")
a.add_argument("--split_cal", default=False, action='store_true', help="Split corrected column in MS from input data.")
a.add_argument("--cal_ext", default="split", type=str, help="Suffix of calibrated MS to split from input data.")
a.add_argument("--split_model", default=False, action='store_true', help="If True, split model column from input data and append model_ext")
a.add_argument("--model_ext", default="model", type=str, help="Suffix of model column in MS to split from input data.")
a.add_argument("--gaintables", default=[], type=str, nargs='*', help="Input gain tables to apply on-the-fly before starting calibration.")

def echo(message, type=0):
    if verbose:
        if type == 0:
            print(message)
        elif type == 1:
            print("\n" + message + "\n" + "-"*40)

if __name__ == "__main__":
    # parse args
    args = a.parse_args()

    # get vars
    verbose = args.silence is False

    # get phase center
    if args.source is not None:
        loc_file = "{}.loc".format(args.source)
        if not os.path.exists(loc_file):
            raise ValueError("A {} file should exist holding its coords given as ha:am:as  deg:am:as".format(loc_file))
        ra, dec = np.loadtxt('{}.loc'.format(args.source), dtype=str)
        ra, dec = ra.split(':'), dec.split(':')
        fixdir = 'J2000 {}h{}m{}s {}d{}m{}s'.format(*(ra+dec))

    msin = args.msin

    # get paths
    base_ms = os.path.basename(msin)
    if args.out_dir is None:
        out_dir = os.path.dirname(msin)
    else:
        out_dir = args.out_dir

    # check for uvfits
    if base_ms.split('.')[-1] == 'uvfits':
        echo("...converting uvfits to ms", type=1)
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
        echo("writing {}".format(msin))
        importuvfits(uvfits, msin)

    # get antenna name to station mapping
    tb.open("{}/ANTENNA".format(msin))
    antstn = tb.getcol("STATION")
    tb.close()
    antstn = [stn for stn in antstn if stn != '']
    antids = [re.findall('\d+', stn)[0] for stn in antstn]
    antid2stn = dict(zip(antids, antstn))

    # configure refant
    if args.refant is None and (args.KGcal is True or args.Acal is True or args.BPcal is True):
        raise AttributeError("if calibrating, refant needs to be specified")
    if args.refant is not None:
        refants = [antid2stn[ra] for ra in args.refant.split(',')]

    # rephase to source
    if args.source is not None:
        echo("...fix vis to {}".format(fixdir), type=1)
        fixvis(msin, msin, phasecenter=fixdir)

    # insert source model
    if args.model is None:
        if args.KGcal is True or args.Acal is True or args.BPcal is True:
            print("...Warning: Asking to calibrate but no model image or component list provided.")
    else:
        if os.path.splitext(args.model)[1] == '.cl':
            echo("...inserting component list {} as MODEL".format(args.model), type=1)
            ft(msin, complist=args.model, usescratch=True)
        else:
            echo("...inserting image {} as MODEL".format(args.model), type=1)
            ft(msin, model=args.model, usescratch=True)

    # unflag
    if args.unflag is True:
        echo("...unflagging", type=1)
        flagdata(msin, mode='unflag')

    # flag autocorrs
    if args.flag_autos:
        echo("...flagging autocorrs", type=1)
        flagdata(msin, autocorr=True)

    # flag bad ants
    if args.ex_ants is not None:
        args.ex_ants = ','.join([antid2stn[xa] for xa in args.ex_ants.split(',')])
        echo("...flagging bad ants: {}".format(args.ex_ants), type=1)
        flagdata(msin, mode='manual', antenna=args.ex_ants)

    # rflag
    if args.rflag is True:
        echo("...rfi flagging", type=1)
        flagdata(msin, mode='rflag')

    def make_cal(cal):
        if args.gain_ext != '':
            c = os.path.join(out_dir, '.'.join([base_ms, args.gain_ext, '{}.cal'.format(cal)]))
        else:
            c = os.path.join(out_dir, '.'.join([base_ms, '{}.cal'.format(cal)]))
        return c

    def KGCAL(msin, gaintables=[]):
        ## perform per-antenna delay and phase calibration ##
        # setup calibration tables    
        kc = make_cal('K')
        gpc = make_cal('Gphs') 

        # perform initial K calibration (per-antenna delay)
        echo("...performing K gaincal", type=1)
        if os.path.exists(kc):
            shutil.rmtree(kc)
        if os.path.exists("{}.png".format(kc)):
            os.remove("{}.png".format(kc))
        # iterate calibration over refants
        for ra in refants:
            gaincal(msin, caltable=kc+'_{}'.format(ra), gaintype="K", solint='inf', refant=ra, minsnr=args.KGsnr,
                    spw=args.gain_spw, gaintable=gaintables, timerange=cal_timerange, uvrange=args.uvrange)
        # merge delay solutions
        kc_files = sorted(glob.glob("{}_*".format(kc)))
        for i, kcf in enumerate(kc_files):
            tb.open(kcf)
            if i == 0:
                delays = tb.getcol('FPARAM')
                delay_ants = tb.getcol('ANTENNA1')
                delay_flags = tb.getcol('FLAG')
                delay_wgts = (~delay_flags).astype(np.float)
            else:
                dlys = tb.getcol('FPARAM')
                dly_ants = tb.getcol('ANTENNA1')
                flgs = tb.getcol('FLAG')
                wgts = (~flgs).astype(np.float)

                delays = (delays*delay_wgts + dlys*wgts) / (delay_wgts + wgts).clip(1e-10, np.inf)
                delay_flags = delay_flags * flgs
                delay_wgts = (~delay_flags).astype(np.float)
            tb.close()
        shutil.copytree(kc_files[0], kc)
        tb.open(kc, nomodify=False)
        tb.putcol("FPARAM", delays)
        tb.putcol("FLAG", delay_flags)
        tb.close()
        for kcf in kc_files:
            shutil.rmtree(kcf)
        # plot cal
        plotcal(kc, xaxis='antenna', yaxis='delay', figfile='{}.png'.format(kc), showgui=False)
        # append to gaintables
        gaintables.append(kc)
        # write delays as npz file
        tb.open(kc)
        delays = tb.getcol('FPARAM')
        delay_ants = tb.getcol('ANTENNA1')
        delay_flags = tb.getcol('FLAG')
        tb.close()
        np.savez("{}.npz".format(kc), delay_ants=delay_ants, delays=delays, delay_flags=delay_flags, shape='(Npol, Nfreq, Nant)')
        echo("...Solved for {} antenna delays".format(np.sum(~delay_flags)))
        echo("...Saving delays to {}.npz".format(kc))
        echo("...Saving plotcal to {}.png".format(kc))

        # perform initial G calibration for phase (per-spw and per-pol gain)
        echo("...performing G gaincal for phase", type=1)
        if os.path.exists(gpc):
            shutil.rmtree(gpc)
        if os.path.exists("{}.png".format(gpc)):
            os.remove("{}.png".format(gpc))
        gaincal(msin, caltable=gpc, gaintype='G', solint='inf', refant=args.refant, minsnr=args.KGsnr, calmode='p',
                spw=args.gain_spw, gaintable=gaintables, timerange=cal_timerange, uvrange=args.uvrange)
        plotcal(gpc, xaxis='antenna', yaxis='phase', figfile='{}.png'.format(gpc), showgui=False)
        gaintables.append(gpc)

        # write phase to file
        tb.open(gpc)
        phases = np.angle(tb.getcol('CPARAM'))
        phase_ants = tb.getcol('ANTENNA1')
        phase_flags = tb.getcol('FLAG')
        tb.close()
        np.savez("{}.npz".format(gpc), phase_ants=phase_ants, phases=phases, phase_flags=phase_flags, shape='(Npol, Nfreq, Nant)')
        echo("...Solved for {} antenna phases".format(np.sum(~phase_flags)))
        echo("...Saving phases to {}.npz".format(gpc))
        echo("...Saving plotcal to {}.png".format(gpc))

        return gaintables

    def ACAL(msin, gaintables=[]):
        # gaincal G amplitude
        echo("...performing G gaincal for amplitude", type=1)
        gac = make_cal("Gamp")

        if os.path.exists(gac):
            shutil.rmtree(gac)
        if os.path.exists("{}.png".format(gac)):
            os.remove("{}.png".format(gac))
        gaincal(msin, caltable=gac, gaintype='G', solint='inf', refant=args.refant, minsnr=args.Asnr, calmode='a',
                spw=args.gain_spw, gaintable=gaintables, timerange=cal_timerange, uvrange=args.uvrange)
        plotcal(gac, xaxis='antenna', yaxis='amp', figfile='{}.png'.format(gac), showgui=False)
        gaintables.append(gac)

        # write amp to file
        tb.open(gac)
        amps = np.abs(tb.getcol('CPARAM'))
        amp_ants = tb.getcol('ANTENNA1')
        amp_flags = tb.getcol('FLAG')
        tb.close()
        np.savez("{}.npz".format(gac), amp_ants=amp_ants, amps=amps, amp_flags=amp_flags, shape='(Npol, Nfreq, Nant)')
        echo("...Solved for {} antenna amps".format(np.sum(~amp_flags)))
        echo("...Saving amps to {}.npz".format(gac))
        echo('...Saving G amp plotcal to {}.png'.format(gac))

        return gaintables

    def BPCAL(msin, gaintables=[]):
        # calibrated bandpass
        echo("...performing B bandpass cal", type=1)
        bc = make_cal("B")

        Btype = "B"
        if args.bpoly:
            Btype="BPOLY"
        if os.path.exists(bc):
            shutil.rmtree(bc)
        if os.path.exists("{}.amp.png".format(bc)):
            os.remove("{}.amp.png".format(bc))
        if os.path.exists("{}.phs.png".format(bc)):
            os.remove("{}.phs.png".format(bc))
        bandpass(vis=msin, spw=args.bp_spw, minsnr=args.BPsnr, bandtype=Btype, degamp=args.degamp, degphase=args.degphase,
                caltable=bc, gaintable=gaintables, solint='inf', refant=args.refant, timerange=cal_timerange,
                uvrange=args.uvrange, solnorm=args.BPsolnorm)
        plotcal(bc, xaxis='chan', yaxis='amp', figfile="{}.amp.png".format(bc), showgui=False)
        plotcal(bc, xaxis='chan', yaxis='phase', figfile="{}.phs.png".format(bc), showgui=False)
        gaintables.append(bc)

        # write bp to file
        if args.bpoly is False:
            # get flags and bandpass data
            tb.open(bc)
            bp = tb.getcol('CPARAM')
            bp_ants = tb.getcol("ANTENNA1")
            bp_flags = tb.getcol('FLAG')
            tb.close()
            # load spectral window data
            tb.open(bc+"/SPECTRAL_WINDOW")
            bp_freqs = tb.getcol("CHAN_FREQ")
            tb.close()
            # write to file
            np.savez("{}.npz".format(bc), bp=bp, bp_ants=bp_ants, bp_flags=bp_flags, bp_freqs=bp_freqs, shape='(Npol, Nfreq, Nant)')
            echo("...Solved for {} antenna bandpasses".format(np.sum(~bp_flags)))
            echo("...Saving bandpass to {}.npz".format(bc))
            echo("...Saving amp plotcal to {}.amp.png".format(bc))
            echo("...Saving phs plotcal to {}.phs.png".format(bc))
        else:
            echo("NOTE: Writing BPOLY bandpass solutions to .npz file not currently supported.")

        return gaintables

    ## Begin Calibration ##
    # init cal_timerange
    cal_timerange = ','.join(args.timerange)

    # run through various calibration options
    gaintables = args.gaintables
    if args.KGcal:
        gaintables = KGCAL(msin, gaintables)

    if args.Acal:
        gaintables = ACAL(msin, gaintables)

    if args.BPcal:
        gaintables = BPCAL(msin, gaintables)

    # apply calibration gaintables
    if len(gaintables) > 0:
        echo("...applying gaintables: \n {}".format('\n'.join(gaintables)), type=1)
        applycal(msin, gaintable=gaintables)

        # split cal
        if args.split_cal:
            # split MS
            ms_split = os.path.join(out_dir, "{}.{}".format(base_ms, args.cal_ext))
            files = glob.glob("{}*".format(ms_split))
            for f in files:
                if os.path.exists(f):
                    try:
                        shutil.rmtree(f)
                    except OSError:
                        os.remove(f)

            echo("...splitting CORRECTED of {} into {}".format(msin, ms_split))
            split(msin, ms_split, datacolumn="corrected")
    else:
        echo("...no calibration performed", type=1)

    if args.split_model:
        # split MS
        ms_split = os.path.join(out_dir, "{}.{}".format(base_ms, args.model_ext))
        files = glob.glob("{}*".format(ms_split))
        for f in files:
            if os.path.exists(f):
                try:
                    shutil.rmtree(f)
                except OSError:
                    os.remove(f)
        echo("...splitting MODEL of {} into {}".format(msin, ms_split))
        split(msin, ms_split, datacolumn="model")


