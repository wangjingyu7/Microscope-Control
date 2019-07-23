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


def get_zernike_indeces_from_args(args):
    if args.z_min > 0 and args.z_max > 0:
        mrange = np.arange(args.z_min, args.z_max + 1)
    else:
        mrange = np.array([], dtype=np.int)

    if args.z_noll_include is not None:
        minclude = np.fromstring(args.z_noll_include, dtype=np.int, sep=',')
        minclude = minclude[minclude > 0]
    else:
        minclude = np.array([], dtype=np.int)

    if args.z_noll_exclude is not None:
        mexclude = np.fromstring(args.z_noll_exclude, dtype=np.int, sep=',')
        mexclude = mexclude[mexclude > 0]
    else:
        mexclude = np.array([], dtype=np.int)

    zernike_indices = np.setdiff1d(
        np.union1d(np.unique(mrange), np.unique(minclude)),
        np.unique(mexclude))

    print('Selected Noll indices for the Zernike modes are:')
    print(zernike_indices)

    if args.z_4pi_modes and args.z_exclude_4pi_modes:
        exclude_4pi = np.fromstring(
            args.z_exclude_4pi_modes, dtype=np.int, sep=',')
        # intersect = np.intersect1d(zernike_indices, np.abs(exclude_4pi))
        # if intersect.size != exclude_4pi.size:
        #     print('WARN: the following 4pi modes exclusions were not used:')
        #     print(np.setdiff1d(exclude_4pi, intersect))
    else:
        exclude_4pi = np.array([], dtype=np.int)

    if args.z_4pi_modes and exclude_4pi.size > 0:
        print('Excluded 4Pi modes are:')
        print(exclude_4pi)

    zernike_indices.sort()
    exclude_4pi.sort()

    return zernike_indices, exclude_4pi


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dm0-index', type=int, default=-1, metavar='INDEX',
        help='CIUsb index for DM0 (the first DM powered up in sequence)')
    parser.add_argument(
        '--dm0-calibration-file', type=str, metavar='HDF5',
        help='Calibration file for DM0')
    parser.add_argument(
        '--dm1-index', type=int, default=-1, metavar='INDEX',
        help='CIUsb index for DM1 (the first DM powered up in sequence)')
    parser.add_argument(
        '--dm1-calibration-file', type=str, metavar='HDF5',
        help='Calibration file for DM1')
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
        '--exclude-4pi-modes', type=str, default='-1,2,3,4',
        metavar='INDICES',
        help='''
Comma separated list of 4Pi modes to ignore, e.g.,
-1,2,3,4 to ignore contravariant piston and covariant tip/tilt/defocus.
The sign denotes co/contra variant. The absolute value denotes a Noll index.
NB: DO NOT USE SPACES in this list!''')
    args = parser.parse_args()

    if args.dm0_index < 0:
        print('Missing --dm0-index', file=sys.err)
        sys.exit()
    if (
            args.dm0_calibration_file is None or
            not path.isfile(args.dm0_calibration_file)):
        print(
            'Missing or wrong path for --dm0-calibration-file',
            file=sys.err)
        sys.exit()
    if args.dm1_index < 0:
        print('Missing --dm1-index', file=sys.err)
        sys.exit()
    if (
            args.dm1_calibration_file is None or
            not path.isfile(args.dm1_calibration_file)):
        print(
            'Missing or wrong path for --dm1-calibration-file',
            file=sys.err)
        sys.exit()

    with File(args.dm0_calibration_file, 'r') as f:
        H0 = f['/WeightedLSCalib/H'][()]
        C0 = f['/WeightedLSCalib/C'][()]
        z0 = f['/WeightedLSCalib/z0'][()]
        n = f['/WeightedLSCalib/cart/RZern/n'][()]
    with File(args.dm1_calibration_file, 'r') as f:
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
        O1 = np.dot(Fy, np.dot(Fx, R))
        C0 = np.dot(O1.T, C0)
        Nz = H0.shape[0]
        assert(Nz == H1.shape[0])
        zernike_indices = np.arange(1, Nz + 1)
        T1 = np.zeros((Nz, Nz))
        all_4pi = list()
        count = 0
        for i, noll in enumerate(zernike_indices):
            for s in [1, -1]:
                all_4pi.append(s*noll)
                T1[i, count] = 1.
                T1[Nz + i, count] = s
                count += 1
        assert(matrix_rank(T1) == 2*Nz)
