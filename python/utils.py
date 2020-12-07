#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 15:56:11 2019

additional functions for ADAN

@author: Lu YI
@organisation: Dept. of EIE, The Hong Kong Polytechnic University
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import os

def leaky_relu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

def norm(X):
    max_ = abs(X).max(axis=0)
    return X/max_, max_

    
def show_result(hist, e, num):
    plt.figure()
    x = range(len(hist['G_losses']))
    
    y1 = hist['D_losses']
    y2 = hist['C_losses']
    y3 = hist['G_losses']
    y4 = hist['R_losses']
    
    plt.plot(x, y1, label = 'Discriminator loss')
    plt.plot(x, y2, label = 'Classifier loss')
    plt.plot(x, y3, label = 'Generator loss')
    plt.plot(x, y4, label = 'Reconstruction loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
  
    plt.legend(fontsize="x-large")
    plt.grid(True)
    plt.tight_layout()

    #plt.show()
    path = './log'
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    f = './log/loss_' + str(e+1) + '_' + str(num) + '.pdf'
    plt.savefig(f)
    plt.close()


def show_result2(hist, e, num):
    plt.figure()
    x = range(len(hist['r_acc']))

    y1 = hist['f_acc']
    y2 = hist['r_acc']
#    y3 = hist['f_std']
#    y4 = hist['r_std']

    plt.plot(x, y1, color='tomato', linestyle='-', linewidth=1.0, label = 'Average probability of synthetic')
    plt.plot(x, y2, color='skyblue', linestyle='-', linewidth=1.0, label = 'Average probability of real')
#    plt.plot(x, y3, color='darkorange', linestyle='-', linewidth=1.0, label = 'Std of probabilities (synthetic)')
#    plt.plot(x, y4, color='g', linestyle='-', linewidth=1.0, label = 'Std of probabilities (real)')
    plt.hlines(0.5, 0, 400, color='lightgray', linewidth=0.5, linestyles='--')

    plt.xlabel('Epoch')
    plt.ylabel('Probability')

    plt.legend(fontsize="x-large")
#    plt.grid(True)
    plt.tight_layout()

    #plt.show()
    path = './log'
    folder = os.path.exists(path)
    if not folder: 
        os.makedirs(path)
    f = './log/acc_' + str(e+1) + '_' + str(num) + '.pdf'
    plt.savefig(f)
    plt.close() 
