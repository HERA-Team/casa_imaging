from pyuvdata import UVData, utils as uvutils
import numpy as np
import argparse
import glob
import os
import sys

ap = argparse.ArgumentParser(description='')

ap.add_argument("filename", type=str, help="Filename to image")
ap.add_argument("model_vis", type=str, help="glob-parseable path to model visibilities")
ap.add_argument("outdir", type=str, help="Output directory to write model file to")

if __name__ == "__main__":

    # parse args
    a = ap.parse_args()

    # load filename metadata
    uvd = UVData()
    uvd.read(a.filename, read_data=False)
    lst_bounds = [uvd.lst_array.min(), uvd.lst_array.max()]
    if lst_bounds[1] < lst_bounds[0]:
        lst_bounds[1] += 2*np.pi

    # get model visibilities
    mfiles = sorted(glob.glob(a.model_vis))
    if len(mfiles) == 0:
        sys.exit(0)

    # get metadata
    mfile_lsts = []
    for mf in mfiles:
        uvm = UVData()
        uvm.read(mf, read_data=False)
        mfile_lsts.append([uvm.lst_array.min(), uvm.lst_array.max()])
    mfile_lsts = np.unwrap(mfile_lsts, axis=0)
    if mfile_lsts[0, 1] < mfile_lsts[0, 0]:
        mfile_lsts[:, 1] += 2*np.pi

    # get files that overlap filename
    model_files = []
    for i, mf_lst in enumerate(mfile_lsts):
        if mf_lst[1] > lst_bounds[0] and mf_lst[0] < lst_bounds[1]:
            model_files.append(mfiles[i])
    if len(model_files) == 0:
        sys.exit(0)

    # load model
    uvm = UVData()
    uvm.read(model_files)

    # down select on lsts
    uvm_lsts = np.unwrap(np.unique(uvm.lst_array))
    tinds = (uvm_lsts >= lst_bounds[0]) & (uvm_lsts <= lst_bounds[1])
    uvm.select(times=np.unique(uvm.time_array)[tinds])

    # write uvfits to outdir
    outname = os.path.basename(filename).replace('uvh5', 'model.uvfits')
    uvm.write_uvfits(os.path.join(a.outdir, outname))
