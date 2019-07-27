#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import json  # noqa
import argparse

from glob import glob
from pprint import pprint
from os import path
from h5py import File
from numpy.linalg import matrix_rank

from czernike import RZern

import matplotlib.pyplot as plt  # noqa


def get_noll_indices(args):
    noll_min = np.array(args.min, dtype=np.int)
    noll_max = np.array(args.max, dtype=np.int)
    minclude = np.array([
        int(s) for s in args.include.split(',')
        if len(s) > 0], dtype=np.int)
    mexclude = np.array([
        int(s) for s in args.exclude.split(',')
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
    assert(np.unique(zernike_indices).size == zernike_indices.size)

    return zernike_indices


def default_name(i, n, m):
    n = f'Z_{i} Z_{n}^{m}'
    if i == 1:
        n += ' piston'
    elif i == 2:
        n += ' tip'
    elif i == 3:
        n += ' tilt'
    elif i == 4:
        n += ' defocus'
    elif m == 0:
        n += ' spherical'
    elif abs(m) == 1:
        n += ' coma'
    elif abs(m) == 2:
        n += ' astigmatism'
    elif abs(m) == 3:
        n += ' trefoil'
    elif abs(m) == 4:
        n += ' quadrafoil'
    elif abs(m) == 5:
        n += ' pentafoil'

    return n


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Make a double objective configuration using 4Pi modes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--flipx', dest='flipx', action='store_true')
    parser.add_argument('--no-flipx', dest='flipx', action='store_false')
    parser.set_defaults(flipx=0)
    parser.add_argument('--flipy', dest='flipy', action='store_true')
    parser.add_argument('--no-flipy', dest='flipy', action='store_false')
    parser.set_defaults(flipy=1)
    parser.add_argument(
        '--rotate', default=0.0, type=float,
        help='Relative pupil rotation in degrees')
    parser.add_argument(
        '--exclude', type=str, default='-1,2,3,4',
        metavar='INDICES',
        help='''
Comma separated list of 4Pi modes to ignore, e.g.,
-1,2,3,4 to ignore contravariant piston and covariant tip/tilt/defocus.
The sign denotes co/contra variant. The absolute value denotes a Noll index.
NB: DO NOT USE SPACES in this list!''')
    parser.add_argument(
        '--include', type=str, default='', metavar='INDICES',
        help='''
Comma separated list of 4Pi modes to include, e.g.,
-1,2,3,4 to ignore contravariant piston and covariant tip/tilt/defocus.
The sign denotes co/contra variant. The absolute value denotes a Noll index.
NB: DO NOT USE SPACES in this list!''')
    parser.add_argument(
        '--min', type=int, default=5,
        help='Minimum Zernike Noll index to consider')
    parser.add_argument(
        '--max', type=int, default=22,
        help='Minimum Zernike Noll index to consider')
    args = parser.parse_args()

    deffiles = glob(path.join('..', '*Default Files*'))[0]
    cfiles = glob(path.join(deffiles, '*.h5'))
    if len(cfiles) != 2:
        print(
            'Leave only *TWO* HDF5 calibration files in the ' +
            'Default Files folder before running this script', file=sys.stderr)
        sys.exit()

    calibs = []
    for c in sorted(cfiles):
        with File(c, 'r') as f:
            wavelength = f['/WeightedLSCalib/wavelength'][()]
            k = wavelength/(2*np.pi)/1000
            H = k*f['/WeightedLSCalib/H'][()]
            C = f['/WeightedLSCalib/C'][()]/k
            d = {
                'H': H,
                'C': C,
                'z': f['/WeightedLSCalib/z0'][()],
                'n': int(f['/WeightedLSCalib/cart/RZern/n'][()]),
                'serial': f['/WeightedLSCalib/dm_serial'][()],
            }
            print(f'DM: {d["serial"]} file: {c}')
            calibs.append(d)

    r = RZern(calibs[0]['n'])
    assert(r.nk == calibs[0]['H'].shape[0])
    assert(r.nk == calibs[1]['H'].shape[0])
    Nz = r.nk

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

    conf = {}

    # serial numbers
    conf['Serials'] = [calibs[0]['serial'], calibs[1]['serial']]
    if conf['Serials'][0] == conf['Serials'][1]:
        print('Error repeated serial number {conf["Serials"]}', sys.stderr)
        sys.exit()

    # flats excluding ttd
    flats = []
    for c in calibs:
        z = c['z']
        C = c['C']

        z[:4] = 0
        flats.append(-np.dot(C, z))

    conf['Flats'] = np.concatenate(flats).tolist()

    # control matrix
    C0 = calibs[0]['C']
    C1 = calibs[1]['C']

    O1 = np.dot(Fy, np.dot(Fx, R))
    C0 = np.dot(C0, O1.T)
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
    all_4pi = np.array(all_4pi, dtype=np.int)

    zernike_indices = get_noll_indices(args)
    print('Selected Zernike indices are:')
    print(zernike_indices)
    print()

    inds = []
    for i in zernike_indices:
        inds.append(np.where(all_4pi == i)[0][0])
    check = all_4pi[inds]
    assert(np.allclose(check, zernike_indices))

    CC = np.zeros((2*C0.shape[0], C0.shape[1] + C1.shape[1]))
    CC[:C0.shape[0], :C0.shape[1]] = C0
    CC[C0.shape[0]:, C0.shape[1]:] = C1
    T3 = np.dot(CC, T1)
    T4 = T3[:, inds]
    conf['Matrix'] = T4.tolist()

    # mode names
    ntab = r.ntab
    mtab = r.mtab
    modes = []
    for i in zernike_indices:
        if i > 0:
            p = 'co-'
        else:
            p = 'contra-'
        n = default_name(i, ntab[abs(i) - 1], mtab[abs(i) - 1])
        modes.append(p + n)

    conf['Modes'] = modes
    print('Selected mode names are:')
    pprint(modes)
    print()

    fname = path.join(deffiles, 'config.json')
    with open(fname, 'w') as f:
        json.dump(conf, f, sort_keys=True, indent=4)
    print('Configuration written to:')
    print(path.abspath(fname))
    print()
