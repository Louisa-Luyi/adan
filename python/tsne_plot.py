#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:24:37 2019

using tsne to plot the real and synthetic samples

@author: Lu YI
@organisation: Dept. of EIE, The Hong Kong Polytechnic University
"""

import scipy.io as sio
import numpy as np
import itertools
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import np_utils
import h5py

plt.switch_backend('agg')

def scatter2D(x, colors, n_colors):
    #sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    # We choose a color palette with seaborn.
    markers = itertools.cycle(('*', '*', 's', 's', '<', '<', 'D','D','X','X','o','o','>','>','p','p','8','8','H','H','v','v'))
    palette = np.array(sns.color_palette("Paired", n_colors))
#    colors_ = ["#a5cee4", "#1677b6", "#b1e086", "#2da122", "#fd9997", "#e61509", "#ffc068", "#ff7f00","#cab1d7","#6b399c","#ffff93","#954616","#d0d0d0","#4f4f4f","#6b9f93","#02562d","#f98473", "#ff0000", "#63dcff"," #0069ff", "#bb79a0", "#83065e"]
#    colors_ = ["#a5cee4", "#1677b6", "#b1e086", "#2da122","#fd9997",#e61509#,"#ffc068","#ff7f00","#cab1d7","6b399c","#ffff93","#954616","#d0d0d0","#4f4f4f"]#,"#6B9F93","#02562D","#F98473","#FF0000","#63DCFF","#0069FF","#BB79A0","#83065E"]
#    palette = np.array(sns.color_palette(colors_))
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    for i in range(n_colors):
        if (i%2==1):
            sc = ax.scatter(x[colors == i,0], x[colors==i, 1], lw=0,s=10,alpha=0.5,
                            c=palette[colors[colors == i].astype(np.int)],marker=markers.__next__())
        else:
            sc = ax.scatter(x[colors == i,0], x[colors==i, 1], lw=0,s=30,alpha=0.5,
                            c=palette[colors[colors == i].astype(np.int)],marker=markers.__next__())
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    ax.axis('tight')
    ax.set_facecolor('white')

    return f, ax, sc

class tsnePlot:
    def __init__(self, path, path_type, times, max_n_points=2000):
        self.path = path
        self.times = times
        self.path_type = path_type
        self.max_n_points = max_n_points

    def plot(self, ref_path, ref_type, output): #output is the file name of the graph, with '.pdf' at the end
        if self.path_type == 'mat':
            data = sio.loadmat(self.path)
            x_fake = np.array(data['x'], dtype='float32')
            y_fake = np.array(data['y'].ravel(), dtype = 'int32')  
        else:
            data = h5py.File(self.path, 'r')
            x_fake = np.array(data['x'][()])
            y_fake = np.array(data['y'][()])
            data.close()
        
        if ref_type == 'mat':
            data_real = sio.loadmat(ref_path)
            x_real = np.array(data_real['x'], dtype='float32')
            y_real = np.array(data_real['y'].ravel(), dtype='int32')
        else:
            data_real = h5py.File(ref_path, 'r')
            x_real = np.array(data_real['x'][()])
            y_real = np.array(data_real['y'][()])
            data_real.close()
        
        y_real_ohe = np_utils.to_categorical(y_real)
        n_colors = 2 * y_real_ohe.shape[1]
        
        for i in range(len(y_fake)):
            y_fake[i] = 2*y_fake[i]+1 
        
        for j in range(len(y_real)):
            y_real[j] = 2*y_real[j]
   
        x = np.concatenate([x_real, x_fake[:int(len(x_fake)/(self.times-1))]], axis=0)
        y = np.concatenate([y_real, y_fake[:int(len(x_fake)/(self.times-1))]], axis=0)

        if x.shape[0] > self.max_n_points:
            idx = np.random.choice(x.shape[0], self.max_n_points, replace=False)
            x = x[idx,:]
            y = y[idx]
        
        x_prj = TSNE().fit_transform(x)
        scatter2D(x_prj, y, n_colors)
        plt.savefig(output)
