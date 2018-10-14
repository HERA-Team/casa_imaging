from __future__ import absolute_import, division, print_function

import numpy as np
import collections
import six
import warnings
from astropy.time import Time
from astropy import coordinates as crd
from astropy import units as unt
import os


def RA2Time(ra, anchor_jd, latitude=-30.72, longitude=21.42, return_lst=True, tolerance=1e-5, maxiter=5):
    """
    Given a right ascension coordinate in J2000 (ICRS), return the
    Local Sidereal Time (LST) or Julian Date on the day of anchor_jd given
    an observer at the specified latitude and longitude on Earth.

    Parameters
    ----------
    ra : float
        Right Ascension in J2000 (ICRS) frame.

    anchor_jd : float
        Julian Date to anchor the conversion from ra to time, because RA wraps with
        respect to time.

    latitude : float
        Latitude on Earth of observer in degrees.

    longitude : float
        Longitude on Earth of observer in degrees.

    return_lst : bool
        If True, return time as local sidereal time in [radian], else return as julian date.

    tolerance : float
        Convergence tolerance in Julian Date for minimizer.

    maxiter : int
        Maximum number of iterations in minimization.
    
    Returns
    -------
    time : float
        LST [radian] or JD when ra is closest to zenith of an observer
    """
    # setup Earth Location
    loc = crd.EarthLocation(lat=latitude * unt.deg, lon=longitude * unt.deg)

    # Enter loop
    k = 0
    time = anchor_jd
    while True:
        # setup Time object with initial Julian Date and a step forward
        t1 = Time(time, format='jd', scale='utc')
        t2 = Time(time+1e-4, format='jd', scale='utc')

        # calculate derivative of d_ra / d_jd
        ra1 = crd.SkyCoord(frame='altaz', alt=90 * unt.deg, az=0 * unt.deg, location=loc, obstime=t1).icrs.ra.degree
        ra2 = crd.SkyCoord(frame='altaz', alt=90 * unt.deg, az=0 * unt.deg, location=loc, obstime=t2).icrs.ra.degree
        d_ra_d_jd = (ra2 - ra1) / 1e-4

        # calculate d_jd from initial time to get desired RA
        if ra < ra1 - 180:
            ra1 -= 360
        d_jd = (ra - ra1) / d_ra_d_jd

        # get new time
        time = time + d_jd

        if np.abs(d_jd) <= tolerance:
            break
        elif k >= maxiter:
            break

        k += 1

    # convert to LST if desired
    if return_lst:
        t = Time(time, format='jd', scale='utc')
        time = t.sidereal_time('apparent', longitude=longitude * unt.deg).radian

    return time

def JD2LST(JD, longitude=21.42830):
    """
    Input:
    ------
    JD : type=float or list of floats containing Julian Date(s) of an observation

    longitude : type=float, longitude of observer in degrees East, default=HERA longitude

    Output:
    -------
    Local Apparent Sidreal Time [radians]

    Notes:
    ------
    The Local Apparent Sidereal Time is *defined* as the right ascension in the current epoch.
    """
    # get JD type
    if isinstance(JD, list) or isinstance(JD, np.ndarray):
        _array = True
    else:
        _array = False
        JD = [JD]

    # iterate over JD
    LST = []
    for jd in JD:
        # construct astropy Time object
        t = Time(jd, format='jd', scale='utc')
        # get LST in radians at epoch of jd
        LST.append(t.sidereal_time('apparent', longitude=longitude * unt.deg).radian)
    LST = np.array(LST)

    if _array:
        return LST
    else:
        return LST[0]


def LST2JD(LST, start_jd, longitude=21.42830):
    """
    Convert Local Apparent Sidereal Time -> Julian Date via a linear fit
    at the 'start_JD' anchor point.

    Input:
    ------
    LST : type=float, local apparent sidereal time [radians]

    start_jd : type=int, integer julian day to use as starting point for LST2JD conversion

    longitude : type=float, degrees East of observer, default=HERA longitude

    Output:
    -------
    JD : type=float, Julian Date(s). accurate to ~1 milliseconds
    """
    # get LST type
    if isinstance(LST, list) or isinstance(LST, np.ndarray):
        _array = True
    else:
        LST = [LST]
        _array = False

    # get start_JD
    base_jd = float(start_jd)

    # iterate over LST
    jd_array = []
    for lst in LST:
        while True:
            # calculate fit
            jd1 = start_jd
            jd2 = start_jd + 0.01
            lst1, lst2 = JD2LST(jd1, longitude=longitude), JD2LST(jd2, longitude=longitude)
            slope = (lst2 - lst1) / 0.01
            offset = lst1 - slope * jd1

            # solve y = mx + b for x
            JD = (lst - offset) / slope

            # redo if JD isn't on starting JD
            if JD - base_jd < 0:
                start_jd += 1
            elif JD - base_jd > 1:
                start_jd -= 1
            else:
                break
        jd_array.append(JD)

    jd_array = np.array(jd_array)

    if _array:
        return jd_array
    else:
        return jd_array[0]


def JD2RA(JD, longitude=21.42830, latitude=-30.72152, epoch='current'):
    """
    Convert from Julian date to Equatorial Right Ascension at zenith
    during a specified epoch.

    Parameters:
    -----------
    JD : type=float, a float or an array of Julian Dates

    longitude : type=float, longitude of observer in degrees east, default=HERA longitude

    latitude : type=float, latitude of observer in degrees north, default=HERA latitutde
               This only matters when using epoch="J2000"

    epoch : type=str, epoch for RA calculation. options=['current', 'J2000'].
            The 'current' epoch is the epoch at JD. Note that
            LST is defined as the zenith RA in the current epoch. Note that
            epoch='J2000' corresponds to the ICRS standard.

    Output:
    -------
    RA : type=float, right ascension [degrees] at zenith JD times
         in the specified epoch.
    """
    # get JD type
    if isinstance(JD, list) or isinstance(JD, np.ndarray):
        _array = True
    else:
        _array = False
        JD = [JD]

    # setup RA list
    RA = []

    # iterate over jd
    for jd in JD:

        # use current epoch calculation
        if epoch == 'current':
            ra = JD2LST(jd, longitude=longitude) * 180 / np.pi
            RA.append(ra)

        # use J2000 epoch
        elif epoch == 'J2000':
            loc = crd.EarthLocation(lat=latitude * unt.deg, lon=longitude * unt.deg)
            t = Time(jd, format='jd', scale='utc')
            zen = crd.SkyCoord(frame='altaz', alt=90 * unt.deg, az=0 * unt.deg, obstime=t, location=loc)
            RA.append(zen.icrs.ra.degree)

        else:
            raise ValueError("didn't recognize {} epoch".format(epoch))

    RA = np.array(RA)

    if _array:
        return RA
    else:
        return RA[0]

