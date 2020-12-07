#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:57:48 2020
@author: mwmak, Dept. of EIE, HKPolyU

Combine the .h5 files in a given folder and save the combined file in .h5 using
virtual datasets.

# Last update: 4 Dec. 2020 by Lu YI
"""
import sys
import pathlib
import h5py
import numpy as np

def combine_h5(filelist, out_h5file):
    n_files = len(filelist)

    n_x = list()
    n_y = list()
    for i in range(n_files):
        with h5py.File(filelist[i], 'r') as f:
            n_x.append((f['x'].shape[0]))
            n_y.append(len(f['y']))
            x_dim = f['x'].shape[1]
    tot_n_x = np.sum(n_x)
    tot_n_y = np.sum(n_y)    
    print(f"Total no. of x = {tot_n_x}")
    print(f"Total no. of y = {tot_n_y}")
    print(f"Feature vectors dim = {x_dim}")

    # Assemble virtual dataset
    x_layout = h5py.VirtualLayout(shape=(tot_n_x, x_dim), dtype=np.float32)
    y_layout = h5py.VirtualLayout(shape=(tot_n_y,), dtype=np.int32)
    k1 = 0
    for i in range(n_files):
        print(f"Reading {filelist[i]}")
        range1 = range(k1, k1 + n_x[i])
        x_layout[list(range1)] = h5py.VirtualSource(filelist[i], "x", shape=(n_x[i], x_dim))
        y_layout[list(range1)] = h5py.VirtualSource(filelist[i], "y", shape=(n_y[i],))
        k1 = k1 + n_x[i]

    # Add virtual dataset to output file
    with h5py.File(out_h5file, "w", libver="latest") as f:
        print(f"Writing combined file {out_h5file}")
        f.create_virtual_dataset("x", x_layout, fillvalue=None)
        f.create_virtual_dataset("y", y_layout, fillvalue=None)

