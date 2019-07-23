#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import json  # noqa
import argparse

from os import path
from h5py import File
from numpy.linalg import matrix_rank

from czernike import RZern


def get_noll_indices(args):
    if args.disable_dm0 or args.disable_dm1:
        raise NotImplementedError()

    noll_min = np.array(args.noll_min, dtype=np.int)
    noll_max = np.array(args.noll_max, dtype=np.int)
    minclude = np.array([
        int(s) for s in args.include_4pi.split(',')
        if len(s) > 0], dtype=np.int)
    mexclude = np.array([
        int(s) for s in args.exclude_4pi.split(',')
        if len(s) > 0], dtype=np.int)
    mrange1 = np.arange(noll_min, noll_max + 1, dtype=np.int)
    mrange = np.zeros(2*mrange1.size)
    mrange[0::2] = mrange1
    mrange[1::2] = -mrange1
    zernike_indices1 = np.setdiff1d(
        np.union1d(np.unique(mrange), np.unique(minclude)),
        np.unique(mexclude))
    zernike_indices = []
    for k in minclude:
        if k in zernike_indices1 and k not in zernike_indices:
            zernike_indices.append(k)
    remaining = np.setdiff1d(zernike_indices1, np.unique(zernike_indices))
    remaining = remaining[np.abs(remaining).argsort()]
    for k in remaining:
        zernike_indices.append(k)
    assert(len(zernike_indices) == zernike_indices1.size)
    zernike_indices = np.array(zernike_indices, dtype=np.int)

    return zernike_indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dm0-calibration', type=str, metavar='HDF5',
        help='Calibration file for DM0 (the first DM powered up in sequence)')
    parser.add_argument(
        '--dm1-calibration', type=str, metavar='HDF5',
        help='Calibration file for DM0 (the second DM powered up in sequence)')
    parser.add_argument(
        '--flipx', action='store_true', help='Set to 1 to flip along x')
    parser.add_argument(
        '--flipy', action='store_true', help='Set to 1 to flip along y')
    parser.add_argument(
        '--rotate', type=float, help='Relative pupil rotation in degrees')
    parser.add_argument(
        '--disable-dm0', action='store_true',
        help='Disable DM0 (single DM control)')
    parser.add_argument(
        '--disable-dm1', action='store_true',
        help='Disable DM1 (single DM control)')
    parser.add_argument(
        '--exclude-4pi', type=str, default='-1,2,3,4',
        metavar='INDICES',
        help='''
Comma separated list of 4Pi modes to ignore, e.g.,
-1,2,3,4 to ignore contravariant piston and covariant tip/tilt/defocus.
The sign denotes co/contra variant. The absolute value denotes a Noll index.
NB: DO NOT USE SPACES in this list!''')
    parser.add_argument(
        '--include-4pi', type=str, default='', metavar='INDICES',
        help='''
Comma separated list of 4Pi modes to include, e.g.,
-1,2,3,4 to ignore contravariant piston and covariant tip/tilt/defocus.
The sign denotes co/contra variant. The absolute value denotes a Noll index.
NB: DO NOT USE SPACES in this list!''')
    parser.add_argument(
        '--noll-min', type=int, default=5,
        help='Minimum Zernike Noll index to consider')
    parser.add_argument(
        '--noll-max', type=int, default=22,
        help='Minimum Zernike Noll index to consider')
    args = parser.parse_args()

    if (
            args.dm0_calibration is None or
            not path.isfile(args.dm0_calibration)):
        print(
            'Missing or wrong path for --dm0-calibration-file',
            file=sys.stderr)
        sys.exit()
    if (
            args.dm1_calibration is None or
            not path.isfile(args.dm1_calibration)):
        print(
            'Missing or wrong path for --dm1-calibration-file',
            file=sys.stderr)
        sys.exit()

    with File(args.dm0_calibration, 'r') as f:
        H0 = f['/WeightedLSCalib/H'][()]
        C0 = f['/WeightedLSCalib/C'][()]
        z0 = f['/WeightedLSCalib/z0'][()]
        n = int(f['/WeightedLSCalib/cart/RZern/n'][()])
    with File(args.dm1_calibration, 'r') as f:
        H1 = f['/WeightedLSCalib/H'][()]
        C1 = f['/WeightedLSCalib/C'][()]
        z1 = f['/WeightedLSCalib/z0'][()]
    r = RZern(n)
    assert(r.nk == H0.shape[0])

    if args.rotate is not None:
        R = r.make_rotation(args.rotate)
    else:
        R = 1

    if args.flipx:
        Fx = r.make_xflip()
    else:
        Fx = 1

    if args.flipy:
        Fy = r.make_yflip()
    else:
        Fy = 1

    if args.disable_dm0 or args.disable_dm1:
        raise NotImplementedError()
    else:
        conf = {}

        z0[:4] = 0
        u0 = -np.dot(C0, z0)
        z1[:4] = 0
        u1 = -np.dot(C1, z1)
        conf['Flats'] = np.concatenate((u0, u1)).tostring()

        O1 = np.dot(Fy, np.dot(Fx, R))
        C0 = np.dot(O1.T, C0)
        Nz = H0.shape[0]
        assert(Nz == H1.shape[0])
        zernike_indices = np.arange(1, Nz + 1)
        T1 = np.zeros((2*Nz, 2*Nz))
        all_4pi = list()
        count = 0
        for i, noll in enumerate(zernike_indices):
            for s in [1, -1]:
                all_4pi.append(s*noll)
                T1[i, count] = 1.
                T1[Nz + i, count] = s
                count += 1
        assert(matrix_rank(T1) == 2*Nz)

        zernike_indices = get_noll_indices(args)
        print('Selected Zernike indices are:')
        print(zernike_indices)
