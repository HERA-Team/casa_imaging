#!/usr/bin/env python2.7
"""
calfits_to_Bcal.py
----------------

A CASA script for converting a
calfits gain file into a CASA
bandpass cal table.

Run as casa -c calfits_to_Bcal.py <args>

Nicholas Kern
Oct. 2018
"""
import pyfits
import os
import shutil
import numpy as np
import argparse

## Set Arguments
a = argparse.ArgumentParser(description="Run with casa as: casa -c calfits_to_Bcal.py <args>")
a.add_argument('--script', '-c', type=str, help='name of this script', required=True)
a.add_argument('--cfits', default=None, type=str, help='Path to calfits FITS file.', required=True)
a.add_argument("--inp_cfile", default=None, type=str, help="Path to CASA bandpass cal table.", required=True)
a.add_argument("--out_cfile", default=None, type=str, help="Name of output cal table. Default is overwrite input.")
a.add_argument("--overwrite", default=False, action='store_true', help='Overwrite output if True.')


def calfits_to_Bcal(cfits, inp_cfile, out_cfile=None, overwrite=False):
    """
    Take a calfits antenna gain file and insert data into an existing
    CASA bandpass calibration table. Note that due to the obtuseness of CASA,
    this function is VERY rigid: the calfits file must be very similar in shape
    and order to the CASA Bcal table.

    It is only recommended to use this script on calfits files that were originally
    *B.cal tables, exported to B.cal.npz files by sky_cal.py and then converted to
    calfits via skynpz2calfits.py.

    Args:
        cfits : str, filepath to pyuvdata calfits file
        inp_cfile : str, filepath to CASA Bandpass calibration table to use as a template
        out_cfile : str, filepath for output CASA Bcal table with cfits data
        overwrite : bool, if True, overwrite output Bcal table
    """
    # assert IO
    if out_cfile is None:
        out_cfile = inp_cfile        
    if os.path.exists(out_cfile) and not overwrite:
        raise IOError("Output cal table {} exists and overwrite is False...".format(out_cfile))

    # move inp_cfile to out_cfile
    if os.path.exists(out_cfile):
        shutil.rmtree(out_cfile)
    shutil.copytree(inp_cfile, out_cfile)

    # load cfits data and get metadata
    hdu = pyfits.open(cfits)
    head = hdu[0].header
    data = hdu[0].data

    # open out_cfile descriptor
    tb.open(out_cfile)
    assert "CPARAM" in tb.getdminfo()['*1']['COLUMNS'], "{} is not a CASA bandpass table...".format(inp_cfile)
    d = tb.getcol("CPARAM")
    f = tb.getcol("FLAG")
    a = tb.getcol("ANTENNA1")

    # The pol axes must match in size
    assert head['NAXIS2'] == d.shape[0], "Npols doesn't match between {} and {}".format(inp_cfile, cfits)

    # real and imag are 0, 1 Image Array axes of fits file
    flags = data[:, 0, :, :, :, 2]
    data = data[:, 0, :, :, :, 0].astype(np.complex) + 1j * data[:, 0, :, :, :, 1]

    # extend to matching antennas
    Nants, Nfreqs, Ntimes, Npols = data.shape
    ants = hdu[1].data['ANTARR'].astype(np.int).tolist()
    _data, _flags = [], []
    for i, ant in enumerate(a):
        if ant in ants:
            aind = ants.index(ant)
            _data.append(data[aind])
            _flags.append(flags[aind])
        else:
            _data.append(np.ones((Nfreqs, Ntimes, Npols), dtype=np.complex))
            _flags.append(np.ones((Nfreqs, Ntimes, Npols), dtype=np.float))
    data = np.asarray(_data, dtype=np.complex)
    flags = np.asarray(_flags, dtype=np.float)    

    # cal table is ordered as ant1_time1, ant2_time1, ... ant1_time2, ant2_time2
    Nants, Nfreqs, Ntimes, Npols = data.shape
    data = np.moveaxis(data, 2, 0).reshape(Nants * Ntimes, Nfreqs, Npols).T
    flags = np.moveaxis(flags, 2, 0).reshape(Nants * Ntimes, Nfreqs, Npols).T

    # now select frequencies that match cal table
    tb.close()
    tb.open("{}/SPECTRAL_WINDOW".format(out_cfile))
    fr = tb.getcol("CHAN_FREQ")[:, 0]
    tb.close()
    freqs = np.arange(head["NAXIS4"]) * head["CDELT4"] + head["CRVAL4"]
    fselect = np.array([np.isclose(_f, fr).any() for _f in freqs])
    data = data[:, fselect, :]
    flags = flags[:, fselect, :]

    # the two arrays must match in shape now
    assert data.shape == d.shape, "fits_data.shape != cal_data.shape..."
    assert flags.shape == f.shape, "fits_flags.shape != cal_flags.shape..."

    # putcol
    print("...inserting {} data and flags into {}".format(cfits, out_cfile))
    tb.open(out_cfile, nomodify=False)
    tb.putcol("CPARAM", data)
    tb.putcol("FLAG", flags)
    tb.close()

    return out_cfile

if __name__ == "__main__":

    # parse args
    args = a.parse_args()
    kwargs = vars(args)
    del kwargs['script']
    cfits = kwargs.pop('cfits')
    inp_cfile = kwargs.pop('inp_cfile')

    # run script
    calfits_to_Bcal(cfits, inp_cfile, **kwargs)

