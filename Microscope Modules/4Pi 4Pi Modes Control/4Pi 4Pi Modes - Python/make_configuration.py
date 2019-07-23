#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import json
import argparse

from h5py import File
from numpy.linalg import norm, matrix_rank

from czernike import RZern


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dm0-index', type=int, help='DM0 CIUsb index', default=-1)
    parser.add_argument(
        '--dm0-calibration-file', type=str, help='DM0 calibration file',
        metavar='HDF5')
    parser.add_argument(
        '--dm1-index', type=int, help='DM1 CIUsb index', default=-1)
    parser.add_argument(
        '--dm1-calibration-file', type=str, help='DM1 calibration file',
        metavar='HDF5')
    parser.add_argument('--flipx', action='store_true')
    parser.add_argument('--flipy', action='store_true')
    parser.add_argument('--rotate', type=float)
    args = parser.parse_args()

    if args.dm0_index < 0:
        print('Missing --dm0-index', file=sys.err)
        sys.exit()
    if (
            args.dm0_calibration_file is None or
            not path.isfile(args.dm0_calibration_file):
        print(f'Wrong file name for --dm0-calibration-file', file=sys.err)
        sys.exit()
    if args.dm1_index < 0:
        print('Missing --dm1-index', file=sys.err)
        sys.exit()
    if (
            args.dm1_calibration_file is None or
            not path.isfile(args.dm1_calibration_file):
        print(f'Wrong file name for --dm1-calibration-file', file=sys.err)
        sys.exit()

    with File(args.dm0_calibration_file, 'r') as f:
        H0 = f['/WeightedLSCalib/H'][()]
        C0 = f['/WeightedLSCalib/C'][()]
        u0 = f['/WeightedLSCalib/uflat'][()]
        n = f['/WeightedLSCalib/cart/RZern/n'][()]
    with File(args.dm1_calibration_file, 'r') as f:
        H1 = f['/WeightedLSCalib/H'][()]
        C1 = f['/WeightedLSCalib/C'][()]
        u1 = f['/WeightedLSCalib/uflat'][()]
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
