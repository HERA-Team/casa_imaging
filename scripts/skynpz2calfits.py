#!/usr/bin/env python2.7
"""
skynpz2calfits.py
---------------

convert sky_cal.py calibration
solution output in .npz format
(originally from CASA .cal/ tables)
into pyuvdata .calfits format.

These gains should be applied to
visibility data using
hera_cal.apply_cal.apply_cal.

Nicholas Kern
August. 2018
"""
import matplotlib
import matplotlib.pyplot as plt
from pyuvdata import UVCal, UVData
import pyuvdata.utils as uvutils
import numpy as np
import argparse
import os
import scipy.signal as signal
from sklearn import gaussian_process as gp
import copy
import hera_cal as hc
import copy

a = argparse.ArgumentParser(description="Turn CASA calibration solutions in {}.npz files from sky_image.py script into .calfits files")

# Required Parameters
a.add_argument("--fname", type=str, help="output calfits filename.", required=True)
a.add_argument("--uv_file", type=str, help="Path to original miriad uv file of data.", required=True)
# Delay Solution Parameters
a.add_argument("--dly_files", type=str, default=None, nargs='*', help="Path to .npz file(s) with antenna delay output from sky_image.py (CASA K cal)")
a.add_argument("--TTonly", default=False, action="store_true", help="only store Tip-Tilt slopes of delay solution.")
a.add_argument("--plot_dlys", default=False, action="store_true", help="plot delay solutions across array.")
# Phase Solution Parameters
a.add_argument("--phs_files", type=str, default=None, nargs='*', help="Path to .npz file(s) with phase output from sky_image.py (CASA G phscal)")
a.add_argument("--plot_phs", default=False, action="store_true", help="plot phase solutions across array.")
# Amplitude Solution Parameters
a.add_argument("--amp_files", type=str, default=None, nargs='*',  help="Path to .npz file(s) with amplitude output from sky_image.py (CASA G ampcal)")
a.add_argument("--plot_amp", default=False, action='store_true', help='plot amp solution across array.')
a.add_argument("--gain_amp_antavg", default=False, action='store_true', help="average gain amplitudes across antennas")
# Bandpass Solution Parameters
a.add_argument("--bp_files", type=str, default=None, nargs='*', help="Path to .npz file(s) with antenna complex bandpass output from sky_image.py (CASA bandpass)")
a.add_argument("--bp_flag_frac", type=float, default=0.5, help="at each freq bin, fraction of antennas flagged needed to broadcast flag to all ants.")
a.add_argument("--bp_broad_flags", default=False, action='store_true', help="broadcast flags at freq channels that satisfy bp_flag_frac")
a.add_argument("--noBPamp", default=False, action='store_true', help="set BP amplitude solutions to zero.")
a.add_argument("--noBPphase", default=False, action='store_true', help="set BP phase solutions to zero.")
a.add_argument('--bp_medfilt', default=False, action='store_true', help="median filter bandpass solutions")
a.add_argument('--medfilt_flag', default=False, action='store_true', help='use median filter to flag bad BP solutions in frequency')
a.add_argument("--medfilt_flag_cut", default=10, type=float, help="If medfilt_flag: number of sigmas to flag channels of medfiltered residual.")
a.add_argument('--medfilt_kernel', default=7, type=int, help="kernel size (channels) for BP median filter")
a.add_argument('--bp_amp_antavg', default=False, action='store_true', help="average bandpass amplitudes across antennas")
a.add_argument('--bp_TTonly', default=False, action='store_true', help="use only tip-tilt phase mode in bandpass solution")
a.add_argument('--plot_bp', default=False, action='store_true', help="plot final bandpass solutions")
a.add_argument('--bp_gp_smooth', default=False, action='store_true', help='smooth bandpass w/ gaussian process. Recommended to precede w/ bp_medfilt.')
a.add_argument('--bp_gp_max_dly', default=200.0, type=float, help="maximum delay in nanosec allowed for gaussian process fit")
a.add_argument('--bp_gp_nrestart', default=1, type=int, help='number of restarts for GP hyperparameter gradient descent.')
a.add_argument('--bp_gp_thin', default=1, type=int, help="thinning factor for freq bins in GP smooth fit")
a.add_argument("--bp_gp_optimizer", default=None, type=str, help="Optimizer to use in GP kernel solution. Pass None for no optimization.")
a.add_argument("--taper_flagged_edges", default=False, action='store_true', help="If GP smoothing and band edges are flagged, connect them to first unflagged channel via smooth exponential taper.")
# Misc
a.add_argument("--out_dir", default=None, type=str, help="output directory for calfits file. Default is working directory path")
a.add_argument("--overwrite", default=False, action="store_true", help="overwrite output calfits file if it exists")
a.add_argument("--multiply_gains", default=False, action="store_true", help="change gain_convention from divide to multiply.")
a.add_argument('--silence', default=False, action='store_true', help="silence output to stdout")

def echo(message, type=0, verbose=True):
    if verbose:
        if type == 0:
            print(message)
        elif type == 1:
            print('\n{}\n{}'.format(message, '-'*40))

def skynpz2calfits(fname, uv_file, dly_files=None, amp_files=None, bp_files=None, out_dir=None, phs_files=None, overwrite=False,
                   TTonly=True, plot_dlys=False, plot_phs=False, gain_convention='multiply', plot_bp=False, noBPphase=False,
                   noBPamp=False, bp_TTonly=False, bp_medfilt=False, medfilt_kernel=5, bp_amp_antavg=False, bp_flag_frac=0.3,
                   bp_broad_flags=False, medfilt_flag=False, bp_gp_smooth=False, bp_gp_max_dly=500.0, bp_gp_nrestart=1, bp_gp_thin=1,
                   plot_amp=False, gain_amp_antavg=False, verbose=True, bp_gp_optimizer=None, medfilt_flag_cut=10,
                   taper_flagged_edges=False):
    """
    Convert *.npz output from sky_image.py into single or multi-pol calfits file.
    Currently only supports data and gain solutions with a single spectral window.
    """
    # get out_dir
    if out_dir is None:
        out_dir = "./"

    # load UVData
    echo("...loading uv_file", verbose=verbose)
    uvd = UVData()
    uvd.read_miriad(uv_file)

    # get ants and antpos
    antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=True)
    ants = list(ants)

    # get freqs, times, jones
    freqs = np.unique(uvd.freq_array)
    times = np.unique(uvd.time_array)
    Nfreqs = len(freqs)
    Nants = len(ants)
    pols = uvd.polarization_array
    jones = np.array([uvutils.jnum2str(p) for p in pols])
    Njones = len(jones)

    # construct blank gains and flags
    gains = np.ones((Nants, Nfreqs, 1, Njones), dtype=np.complex)
    flags = np.zeros((Nants, Nfreqs, 1, Njones), dtype=np.bool)
    flagged_ants = np.zeros(Nants, dtype=np.bool)

    # process delays if available
    if dly_files is not None:
        echo("...processing delays", verbose=verbose, type=1)
        delays = np.zeros((Nants, Njones), dtype=np.float)
        delay_flags = np.zeros((Nants, Njones), dtype=np.bool)
        for dly_file in dly_files:
            echo("...processing {}".format(dly_file), verbose=verbose)
            # get CASA delays and antennas
            dly_data = np.load(dly_file)
            dly_ants = dly_data['delay_ants']
            dlys = dly_data['delays'][:Njones, :]
            dly_flags = dly_data['delay_flags'][:Njones, :]
            dly_ants = dly_ants.tolist()
            dlys *= 1e-9
            # reorder antennas to be consistant w/ ants array
            dlys = np.array([dlys[:, dly_ants.index(a)] if a in dly_ants else 0.0 for a in ants])
            dly_flags = np.array([dly_flags[:, dly_ants.index(a)] if a in dly_ants else True for a in ants])
            # keep only limited information
            if TTonly:
                echo("...keeping only TT component of delays", verbose=verbose)
                A = np.vstack([antpos[:, 0], antpos[:, 1]]).T
                fit = np.linalg.pinv(A.T.dot(A)).dot(A.T).dot(dlys)
                dlys = A.dot(fit)
            # add to delay array
            delays += dlys
            delay_flags += dly_flags
            flagged_ants += np.min(dly_flags, axis=1)  # only a flagged ant if all pols are flagged

        # turn into complex gains
        dly_gains = np.exp(2j*np.pi*(freqs-freqs.min())[None, :, None, None] * delays[:, None, None, :])
        dly_gain_flags = delay_flags[:, None, None]

        # multiply into gains
        gains *= dly_gains

        # add into flags
        flags += dly_gain_flags

    # process overall phase if available
    if phs_files is not None:
        echo("...processing gain phase", verbose=verbose, type=1)
        phases = np.zeros((Nants, Njones), dtype=np.float)
        phase_flags = np.zeros((Nants, Njones), dtype=np.bool)
        for phs_file in phs_files:
            echo("...processing {}".format(phs_file), verbose=verbose)
            # get phase antennas and phases
            phs_data = np.load(phs_file)
            phs_ants = phs_data['phase_ants']
            phs = phs_data['phases'][:Njones, :]
            phs_flags = phs_data['phase_flags'][:Njones, :]
            phs_ants = phs_ants.tolist()
            # reorder to match ants
            phs = np.array([phs[:, phs_ants.index(a)] if a in phs_ants else 0.0 for a in ants])
            phs_flags = np.array([phs_flags[:, phs_ants.index(a)] if a in phs_ants else True for a in ants])
            # add to phases
            phases += phs
            phase_flags += phs_flags
            flagged_ants += np.min(phs_flags, axis=1)

        # construct gains
        phase_gains = np.exp(1j * phases)

        # mult into gains
        gains *= phase_gains[:, None, None]

        # add into flags
        flags += phase_flags[:, None, None]

    # process overall amplitude if available
    if amp_files is not None:
        echo("...processing gain amp", verbose=verbose, type=1)
        amplitudes = np.ones((Nants, Njones), dtype=np.float)
        amplitude_flags = np.zeros((Nants, Njones), dtype=np.bool)
        for amp_file in amp_files:
            echo("...processing {}".format(amp_file), verbose=verbose)
            # get amp antenna and amps
            amp_data = np.load(amp_file)
            amp_ants = amp_data['amp_ants']
            amps = amp_data['amps'][:Njones, :]
            amp_flags = amp_data['amp_flags'][:Njones, :]
            amp_ants = amp_ants.tolist()
            # reorder to match ants
            amps = np.array([amps[:, amp_ants.index(a)] if a in amp_ants else 1.0 for a in ants])
            amp_flags = np.array([amp_flags[:, amp_ants.index(a)] if a in amp_ants else True for a in ants])
            # add to amplitudes
            amplitudes *= amps
            amplitude_flags += amp_flags
            flagged_ants += np.min(amp_flags, axis=1)

        # average across ants if desired
        if gain_amp_antavg:
            echo("...averaging antenna gain amplitudes", verbose=verbose)
            avg_amp = np.median(amplitudes[~amplitude_flags])
            amplitudes *= avg_amp / amplitudes

        # mult into gains
        gains *= amplitudes[:, None, None, :]

        # add into flags
        flags += amplitude_flags[:, None, None, :]

    # process bandpass if available
    if bp_files is not None:
        echo("...processing bandpass", verbose=verbose, type=1)

        bp_gains = np.ones_like(gains)
        bp_flags = np.zeros_like(flags)

        for ii, bp_file in enumerate(bp_files):
            echo("...processing {}".format(bp_file), verbose=verbose)
            # get bandpass and form complex gains
            bp_data = np.load(bp_file)
            bp_freqs = bp_data['bp_freqs']
            bp_Nfreqs = len(bp_freqs)
            bandp_gains = bp_data['bp'][:Njones, :, :]
            bandp_flags = bp_data['bp_flags'][:Njones, :, :]
            bp_ants = bp_data['bp_ants'].tolist()

            # reorder to match ants
            bandp_gains = np.array([bandp_gains[:, :, bp_ants.index(a)].T if a in bp_ants else np.zeros((bp_Nfreqs), dtype=np.complex) for a in ants])
            bandp_flags = np.array([bandp_flags[:, :, bp_ants.index(a)].T if a in bp_ants else np.ones((bp_Nfreqs), dtype=np.bool) for a in ants])

            # broadcast flags to all antennas at freq channels that satisfy bp_flag_frac
            flag_broadcast = (np.sum(bandp_flags, axis=0) / float(bandp_flags.shape[0])) > bp_flag_frac
            if bp_broad_flags:
                bandp_flags += np.repeat(flag_broadcast[None, :, :].astype(np.bool), Nants, 0)

            # configure gains and flags shapes
            bandp_gains = bandp_gains[:, :, None]
            bandp_flags = bandp_flags[:, :, None]

            # figure out which frequencies match
            match_freq = np.zeros_like(freqs, dtype=np.bool)
            match_freq[[np.argmin(np.abs(freqs-f)) for f in bp_freqs]] = True

            # multipy into bp_gains: flag non-overlapping channels
            bp_gains[:, match_freq, :, :] *= bandp_gains
            bp_flags[:, match_freq, :, :] += bandp_flags
            bp_flags[:, ~match_freq, :, :] = True

        # set flagged pixels to zero in bandpass
        bp_gains[bp_flags] = 0.0

        # median filter if desired
        if bp_medfilt or medfilt_flag:
            echo("...median filtering", verbose=verbose)
            bp_gains_medfilt = signal.medfilt(bp_gains.real, kernel_size=(1, medfilt_kernel, 1, 1)) + 1j*signal.medfilt(bp_gains.imag, kernel_size=(1, medfilt_kernel, 1, 1))

            if medfilt_flag:
                echo("...solving for BP flags w/ medfilt data", verbose=verbose)
                # get residual and MAD from unfiltered and filtered data
                residual = (np.abs(bp_gains) - np.abs(bp_gains_medfilt))
                residual[bp_flags] *= np.nan
                resid_std = np.nanmedian(np.abs(residual - np.nanmedian(residual, axis=1, keepdims=True)), axis=1, keepdims=True) * 1.5
                # identify outliers as greater than 10-sigma
                bad = np.array([np.abs(residual[i]) > resid_std[i]*medfilt_flag_cut for i in range(residual.shape[0])])
                bp_flags += bad

            if bp_medfilt:
                echo("...bandpass is the median filtered bandpass", verbose=verbose)
                bp_gains = bp_gains_medfilt

        # average amplitude across antenna if desired
        if bp_amp_antavg:
            echo("...averaging bandpass amplitude across antennas", verbose=verbose)
            amp_avg = np.nanmedian(np.abs(bp_gains), axis=0, keepdims=True)
            bp_gains *= amp_avg / np.abs(bp_gains)

        # smooth bandpass w/ gaussian process if desired
        _bp_gains = bp_gains.copy()
        if bp_gp_smooth:
            echo("...smoothing with gaussian process", verbose=verbose)
            freq_lambda = 1. / (bp_gp_max_dly*1e-3) # MHz
            kernel = 1**2 * gp.kernels.RBF(freq_lambda, (freq_lambda, 200.0)) + gp.kernels.WhiteKernel(1e-4, (1e-8, 1e0))

            # configure data
            X = freqs[:, None] / 1e6
            bp_gains_real = []
            bp_gains_imag = []
            if taper_flagged_edges:
                echo("...tapering flagged band edges", verbose=verbose)
                if bp_gp_max_delay < 500:
                    echo("...Warning: using taper_flagged_edges with a low max delay cut (<500 ns) can throw off solution at band edges in certain cases.")

            # iterate over antennas
            for i, a in enumerate(ants):
                if i % 10 == 0: echo("{}/{} ants".format(i, len(ants)), verbose=verbose)
                ant_gain_real = []
                ant_gain_imag = []
                for j, p in enumerate(jones):
                    # skip flagged ants
                    if np.min(bp_flags[i, :, :, j]):
                        ant_gain_real.append(np.zeros_like(X.squeeze()))
                        ant_gain_imag.append(np.zeros_like(X.squeeze()))
                        continue

                    # Setup Gaussian Process Regressor
                    GP = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=bp_gp_nrestart, optimizer=bp_gp_optimizer)

                    # Taper flagged edges if desired
                    flagged = bp_flags[i, :, 0, j].copy()
                    if taper_flagged_edges:
                        # for band edges that are completely flagged
                        # connect them to first unflagged channel via
                        # a smooth exponential taper
                        def taper(f, g):
                            start = np.argmin(f)
                            N = start * 2 + 1
                            t = signal.windows.exponential(N, tau=N/20.0)[:N//2]
                            t /= t.max()
                            g[:start] = t * g[start]
                            f[:start] = False
                        taper(flagged, bp_gains[i, :, 0, j])
                        taper(flagged[::-1], bp_gains[i, ::-1, 0, j])

                    # Get unflagged frequencies X and Y data for this ant-pol
                    unflagged = ~flagged
                    xdata = X[unflagged][::bp_gp_thin]
                    yreal = bp_gains[i, unflagged, 0, j][::bp_gp_thin].real
                    yimag = bp_gains[i, unflagged, 0, j][::bp_gp_thin].imag

                    # stack real and imag into separate features
                    ydata = np.vstack([yreal, yimag]).T

                    # fit for GP covariance
                    GP.fit(xdata, ydata)

                    # make predictions across full freq band and then add ymedian back in
                    ypred = GP.predict(X)

                    # append
                    ant_gain_real.append(ypred[:, 0])
                    ant_gain_imag.append(ypred[:, 1])
                # append
                bp_gains_real.append(ant_gain_real)
                bp_gains_imag.append(ant_gain_imag)
            # reconstruct bp gains
            bp_gains_real = np.moveaxis(bp_gains_real, 1, 2)[:, :, None, :]
            bp_gains_imag = np.moveaxis(bp_gains_imag, 1, 2)[:, :, None, :]
            bp_gains = bp_gains_real.astype(np.complex) + 1j*bp_gains_imag

        # take only tip-tilt if desired
        if bp_TTonly:
            raise NotImplementedError("bp_TTonly not fully implemented...")
            echo("...distilling bandpass to only tip-tilt phase", verbose=verbose)
            # get bandpass phase array
            bp_phs = np.angle(bp_gains)
            # form least squares estimate of phase slopes along X and Y
            A = np.vstack([antpos[:, 0], antpos[:, 1]]).T
            projection = np.linalg.pinv(A.T.dot(A)).dot(A.T)
            fit = np.einsum("ij,jklm", projection, bp_phs)
            # make prediction to get tip-tilt estimates
            bp_TT_phs = np.einsum("ji,iklm", A, fit)
            # update bandpass gains
            bp_gains *= np.exp(1j*bp_TT_phs - 1j*np.angle(bp_gains))

        # suppress amp and/or phase if desired
        if noBPamp:
            echo("...eliminating BP amplitudes", verbose=verbose)
            bp_gains /= np.abs(bp_gains)

        if noBPphase:
            echo("...eliminating BP phases", verbose=verbose)
            bp_gains /= np.exp(1j*np.angle(bp_gains))

        # mult into gains: _ denotes unsmoothed if smoothed is available
        _gains = gains * _bp_gains
        gains = gains * bp_gains
        flags += bp_flags
        bp_flagged_ants = np.min(bp_flags, axis=(1, 2, 3))
        flagged_ants += bp_flagged_ants

    # check filename
    if fname.split('.')[-1] != "calfits":
        fname += ".calfits"

    # make dictionaries
    gain_dict = {}
    flag_dict = {}
    for i, a in enumerate(ants):
        for j, p in enumerate(jones):
            gain_dict[(a, p)] = gains[i, :, :, j].T.conj()
            flag_dict[(a, p)] = flags[i, :, :, j].T
            if flagged_ants[i]:
                flag_dict[(a, p)] += True

    # write to calfits
    uvc = hc.io.write_cal(fname, gain_dict, freqs, times[:1], flags=flag_dict, outdir=out_dir,
                          overwrite=overwrite, gain_convention=gain_convention, zero_check=True)

    # plot dlys
    if plot_dlys:
        fig, axes = plt.subplots(Njones, figsize=(8,12))
        if Njones == 1:
            axes = [axes]
        for i, j in enumerate(jones):
            ax = axes[i]
            ax.grid(True)
            dly_max = np.max(np.abs(delays[:, i]*1e9))
            dly_min = -dly_max
            for k, a in enumerate(ants):
                if flagged_ants[k] == True:
                    continue
                cax = ax.scatter(antpos[k, 0], antpos[k, 1], c=delays[k, i]*1e9, s=200, cmap='coolwarm', vmin=dly_min, vmax=dly_max)
                ax.text(antpos[k, 0] + 1, antpos[k, 1] + 2, str(a), fontsize=12)
            cbar = fig.colorbar(cax, ax=ax)
            cbar.set_label("delay [nanosec]", size=16)
            ax.set_xlabel("X [meters]", fontsize=14)
            ax.set_ylabel("Y [meters]", fontsize=14)
            ax.set_title("{} Delay solutions for {}".format(j, os.path.basename(dly_files[0])), fontsize=10)
        pname = dly_files[-1]+'.png'
        echo("...saving {}".format(pname), verbose=verbose)
        fig.savefig(pname, dpi=150, bbox_inches='tight', pad=0.05)
        plt.close()

    # plot phs
    if plot_phs:
        fig, axes = plt.subplots(Njones, figsize=(8,12))
        if Njones == 1:
            axes = [axes]
        for i, j in enumerate(jones):
            ax = axes[i]
            ax.grid(True)
            phs_max = np.pi
            phs_min = -np.pi
            for k, a in enumerate(ants):
                if flagged_ants[k] == True:
                    continue
                cax = ax.scatter(antpos[k, 0], antpos[k, 1], c=phases[k, i], s=200, cmap='viridis', vmin=phs_min, vmax=phs_max)
                ax.text(antpos[k, 0] + 1, antpos[k, 1] + 2, str(a), fontsize=12) 
            cbar = fig.colorbar(cax, ax=ax)
            cbar.set_label("phase [radians]", size=16)
            ax.set_xlabel("X [meters]", fontsize=14)
            ax.set_ylabel("Y [meters]", fontsize=14)
            ax.set_title("{} Phase solutions for {}".format(j, os.path.basename(phs_files[0])), fontsize=10)
        pname = phs_files[-1]+'.png'
        echo("...saving {}".format(pname), verbose=verbose)
        fig.savefig(pname, dpi=150, bbox_inches='tight', pad=0.05)
        plt.close()

    # plot amp
    if plot_amp:
        fig, axes = plt.subplots(Njones, figsize=(8,12))
        if Njones == 1:
            axes = [axes]
        for i, j in enumerate(jones):
            ax = axes[i]
            ax.grid(True)
            amp_med = np.nanmedian(amplitudes[~amplitude_flags[:, i], i])
            amp_std = np.std(amplitudes[~amplitude_flags[:, i], i])
            amp_max = amp_med + amp_std * 2
            amp_min = amp_med - amp_std * 2
            for k, a in enumerate(ants):
                if flagged_ants[k] == True:
                    continue
                cax = ax.scatter(antpos[k, 0], antpos[k, 1], c=amplitudes[k, i], s=200, cmap='rainbow', vmin=amp_min, vmax=amp_max)
                ax.text(antpos[k, 0] + 1, antpos[k, 1] + 2, str(a))
            cbar = fig.colorbar(cax, ax=ax)
            cbar.set_label("amplitude", size=16)
            ax.set_xlabel("X [meters]", fontsize=14)
            ax.set_ylabel("Y [meters]", fontsize=14)
            ax.set_title("{} amplitude solutions for {}".format(j, os.path.basename(amp_files[0])), fontsize=10)
        pname = amp_files[-1]+'.png'
        echo("...saving {}".format(pname), verbose=verbose)
        fig.savefig(pname, dpi=150, bbox_inches='tight', pad=0.05)
        plt.close()

    # plot bandpass
    if plot_bp:
        def plot_bandpass(gains, title, pname):
            fig, axes = plt.subplots(2, Njones, figsize=(14, 6))
            if Njones == 1:
                axes.resize(2, 1)
            for i, j in enumerate(jones):
                axs = axes[:, i]
                fig.subplots_adjust(hspace=0.3)
                # amplitude
                ax = axs[0]
                ax.grid(True)
                gains[flags] *= np.nan
                pls = []
                bp_ant_select = []
                ant_sort = np.argsort(ants)
                for k, a in enumerate(np.array(ants)[ant_sort]):
                    if flagged_ants[ant_sort][k] == True:
                        continue
                    p, = ax.plot(freqs / 1e6, np.abs(gains)[ants.index(a), :, 0, i], marker='.', ls='', markersize=1)
                    pls.append(p)
                ax.set_xlabel("Frequency [MHz]", fontsize=12)
                ax.set_ylabel("Amplitude", fontsize=12)
                ax.set_title("{} Total Gain for {}".format(title, j), fontsize=10)
                # phase
                ax = axs[1]
                ax.grid(True)
                plot_ants = []
                for k, a in enumerate(np.array(ants)[ant_sort]):
                    if flagged_ants[ant_sort][k] == True:
                        continue
                    plot_ants.append(a)
                    ax.plot(freqs / 1e6, np.angle(gains)[ants.index(a), :, 0, i], marker='.', ls='', markersize=1)
                ax.set_xlabel("Frequency [MHz]", fontsize=12)
                ax.set_ylabel("Phase [radians]", fontsize=12)
                lax = fig.add_axes([1.01, 0.1, 0.05, 0.8])
                lax.axis('off')
                lax.legend(pls, plot_ants, ncol=2)
            echo("...saving {}".format(pname), verbose=verbose)
            fig.savefig(pname, dpi=150, bbox_inches='tight', pad=0.05)
            plt.close()

        # plot full bandpass
        plot_bandpass(copy.deepcopy(gains), '', os.path.splitext(fname)[0]+'.png')
        if bp_gp_smooth:
            # plot residual with unsmoothed
            plot_bandpass(copy.deepcopy(gains/_gains), "Smoothed / Unsmoothed", os.path.splitext(fname)[0]+'.resid.png')


if __name__ == "__main__":
    args = a.parse_args()

    gain_convention = 'divide'
    if args.multiply_gains:
        gain_convention = 'multiply'

    kwargs = copy.copy(args).__dict__
    kwargs.pop('fname')
    kwargs.pop('uv_file')
    kwargs.pop('multiply_gains')
    kwargs['verbose'] = args.silence == False
    kwargs.pop('silence')
    kwargs['gain_convention'] = gain_convention

    skynpz2calfits(args.fname, args.uv_file, **kwargs)

