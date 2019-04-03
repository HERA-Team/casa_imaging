#!/usr/bin/env python2.7
"""
source2file.py
=============

Given a calibrator source,
output the files that contain
when it is closest to zenith
"""
import os
import numpy as np
import argparse
from astropy.time import Time
from pyuvdata import UVData
import sys
import casa_imaging.coord_convs as cc

ap = argparse.ArgumentParser(description='')

ap.add_argument("--ra", type=float, help="RA of the source in degrees (J2000)", required=True)
ap.add_argument("--lon", default=21.428305555, type=float, help="longitude of observer on Earth in degrees East")
ap.add_argument("--start_jd", type=int, help="starting JD of interest")
ap.add_argument("--duration", default=2.0, type=float, help="duration in minutes of calibrator integration")
ap.add_argument("--offset", default=0.0, type=float, help="offset from closest approach in minutes")
ap.add_argument("--jd_files", default=None, type=str, nargs='*', help="glob-parsable search of files to isolate calibrator within.")
ap.add_argument("--get_filetimes", default=False, action='store_true', help="open source files and get more accurate duration timerange")


def echo(message, type=0, verbose=True):
    if verbose:
        if type == 0:
            print(message)
        elif type == 1:
            print('\n{}\n{}'.format(message, '-'*70))


def source2file(ra, lon=21.428305555, lat=-30.72152, duration=2.0, offset=0.0, start_jd=None,
                    jd_files=None, get_filetimes=False, verbose=False):
    """
    """
    # get LST of source
    # LEGACY: lst = RA2LST(ra, lon, lat, start_jd)
    lst = cc.RA2Time(ra, start_jd, longitude=lon, latitude=lat, return_lst=True) * 12.0 / np.pi

    # offset
    lst += offset / 60.

    echo("source LST (offset by {} minutes) = {} Hours".format(offset, lst), type=1, verbose=verbose)

    jd = None
    utc_range = None
    utc_center = None
    source_files = None
    source_utc_range = None

    # get JD when source is at zenith
    jd = cc.LST2JD(lst * np.pi / 12., start_jd, longitude=lon)
    echo("JD closest to zenith (offset by {} minutes): {}".format(offset, jd), type=1, verbose=verbose)

    # print out UTC time
    jd_duration = duration / (60. * 24 + 4.0)
    time1 = Time(jd - jd_duration/2, format='jd').to_datetime()
    time2 = Time(jd + jd_duration/2, format='jd').to_datetime()
    time3 = Time(jd, format='jd').to_datetime()
    utc_range = '"{:04d}/{:02d}/{:02d}/{:02d}:{:02d}:{:02d}~{:04d}/{:02d}/{:02d}/{:02d}:{:02d}:{:02d}"'\
                ''.format(time1.year, time1.month, time1.day, time1.hour, time1.minute, time1.second,
                          time2.year, time2.month, time2.day, time2.hour, time2.minute, time2.second)
    utc_center = '{:04d}/{:02d}/{:02d}/{:02d}:{:02d}:{:02d}'.format(time3.year, time3.month, time3.day,
                                                                    time3.hour, time3.minute, time3.second)
    echo('UTC time range of {} minutes is:\n{}\ncentered on {}'\
         ''.format(duration, utc_range, utc_center), type=1, verbose=verbose)

    if jd_files is not None:

        # get files
        files = jd_files
        if len(files) == 0:
            raise AttributeError("length of jd_files is zero")

        # keep files with start_JD in them
        file_jds = []
        for i, f in enumerate(files):
            if str(start_jd) not in f:
                files.remove(f)
            else:
                fjd = os.path.basename(f).split('.')
                findex = fjd.index(str(start_jd))
                file_jds.append(float('.'.join(fjd[findex:findex+2])))
        files = np.array(files)[np.argsort(file_jds)]
        file_jds = np.array(file_jds)[np.argsort(file_jds)]

        # get file with closest jd1 that doesn't exceed it
        jd1 = jd - jd_duration / 2
        jd2 = jd + jd_duration / 2

        jd_diff = file_jds - jd1
        jd_before = jd_diff[jd_diff < 0]
        if len(jd_before) == 0:
            start_index = np.argmin(np.abs(jd_diff))
        else:
            start_index = np.argmax(jd_before) 

        # get file closest to jd2 that doesn't exceed it
        jd_diff = file_jds - jd2
        jd_before = jd_diff[jd_diff < 0]
        if len(jd_before) == 0:
            end_index = np.argmin(np.abs(jd_diff))
        else:
            end_index = np.argmax(jd_before) 

        source_files = files[start_index:end_index+1]

        echo("file(s) closest to source (offset by {} minutes) over {} min duration:\n {}"\
             "".format(offset, duration, source_files), type=1, verbose=verbose)

        if get_filetimes:
            # Get UTC timerange of source in files
            uvd = UVData()
            for i, sf in enumerate(source_files):
                if i == 0:
                    uvd.read(sf)
                else:
                    uv = UVData()
                    uv.read(sf)
                    uvd += uv
            file_jds = np.unique(uvd.time_array)
            file_delta_jd = np.median(np.diff(file_jds))
            file_delta_min =  file_delta_jd * (60. * 24)
            num_file_times = int(np.ceil(duration / file_delta_min))
            file_jd_indices = np.argsort(np.abs(file_jds - jd))[:num_file_times]
            file_jd1 = file_jds[file_jd_indices].min()
            file_jd2 = file_jds[file_jd_indices].max()

            time1 = Time(file_jd1, format='jd', scale='utc').to_datetime()
            time2 = Time(file_jd2, format='jd', scale='utc').to_datetime()
            time3 = Time(file_jd1 + jd_duration/2.0, format='jd', scale='utc').to_datetime()

            source_utc_range = '"{:04d}/{:02d}/{:02d}/{:02d}:{:02d}:{:02d}~{:04d}/{:02d}/{:02d}/{:02d}:{:02d}:{:02d}"'\
                               ''.format(time1.year, time1.month, time1.day, time1.hour, time1.minute, time1.second,
                                         time2.year, time2.month, time2.day, time2.hour, time2.minute, time2.second)
            source_utc_center = '{:04d}/{:02d}/{:02d}/{:02d}:{:02d}:{:02d}'.format(time3.year, time3.month, time3.day,
                                                                            time3.hour, time3.minute, time3.second)

            echo('UTC time range of source in files above over {} minutes is:\n{}\ncentered on {} = {}'\
                 ''.format(duration, source_utc_range, source_utc_center, file_jd1 + jd_duration/2.0),
                 type=1, verbose=verbose) 

    return (lst, jd, utc_range, utc_center, source_files, source_utc_range)

if __name__ == "__main__":
    # parse arge
    a = ap.parse_args()
    kwargs = dict(vars(a))
    ra = a.ra
    kwargs.pop('ra')
    kwargs['verbose'] = True
    output = source2file(ra, **kwargs)

