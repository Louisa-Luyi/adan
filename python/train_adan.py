
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:00:16 2019

Training the ADAN

@author: Lu YI
@organisation: Dept. of EIE, The Hong Kong Polytechnic University
"""

from __future__ import print_function

import scipy.io as sio
import tensorflow as tf
import numpy as np
from adan import ADAN
from tsne_plot import tsnePlot
import os
import h5py
from combine_h5 import combine_h5

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("dataset", "emodb", "dataset to be used, either emodb or iemocap")
tf.app.flags.DEFINE_string("loso", "spk", "leave-one-speaker out or leave-one-session-out (only for iemocap)") #spk or sess
tf.app.flags.DEFINE_string("data_dir", "full", "using full data or improvised only for iemocap")
tf.app.flags.DEFINE_integer("latent_dim", 100, "dimension of encoder's output and generator's output")
tf.app.flags.DEFINE_integer("z_dim", 100, "dimension of generator's input")
tf.app.flags.DEFINE_float("lr_c", 0.0001, "learning rate of the classifier")
tf.app.flags.DEFINE_float("lr_d", 0.0001, "learning rate of the discriminator")
tf.app.flags.DEFINE_float("lr_g", 0.0001, "learning rate of the generator")
tf.app.flags.DEFINE_float("lr_e", 0.0001, "learning rate of the encoder")
tf.app.flags.DEFINE_float("lr_r", 0.0001, "learning rate of the decoder")
tf.app.flags.DEFINE_integer("n_times", 10, "the ratio: (synthetic samples)/(original samples)")
tf.app.flags.DEFINE_integer("d_times", 2, "the number of training times of the discriminator in each iteration")
tf.app.flags.DEFINE_integer("e_times", 1, "the number of training times of the encoder in each iteration")
tf.app.flags.DEFINE_bool("imbalanced", False, "whether the dataset is imbalanced")
tf.app.flags.DEFINE_integer("batch", 128, "batch size")
tf.app.flags.DEFINE_integer("epoch", 1500, "training epochs")
tf.app.flags.DEFINE_string("distance", "adversarial", "using adversarial learning or Wassestein divergence")
tf.app.flags.DEFINE_bool("load_mode", False, "whether load existing model")
tf.app.flags.DEFINE_string("model", "./model_adan/model_loso1_final.ckpt", "folder to load the model")
tf.app.flags.DEFINE_string("res_path", f"../result/{FLAGS.dataset}_adan", "folder to store the generated samples")
tf.app.flags.DEFINE_bool("plot", True, "whether plot the results using tSNE")

def augment(res_path, trn, tst, syn_path, num_loso):

    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if FLAGS.distance == 'Wasserstein':
            adan = ADAN(trn, sess, latent_dim=FLAGS.latent_dim, z_dim=FLAGS.z_dim,
                        lr_C=FLAGS.lr_c, lr_D=FLAGS.lr_d, lr_G=FLAGS.lr_g,
                        lr_E=FLAGS.lr_e, lr_R=FLAGS.lr_g, times=FLAGS.n_times,
                        d_times=FLAGS.d_times, e_times=FLAGS.e_times,
                        imbalanced=FLAGS.imbalanced, batch=FLAGS.batch,
                        epoch=FLAGS.epoch, num=num_loso)

            if not FLAGS.load_mode:
                adan.train_wadan()
            else:
                adan.load_wadan(FLAGS.model)

        else:
            adan = ADAN(trn, sess, latent_dim=FLAGS.latent_dim, z_dim=FLAGS.z_dim,
                        lr_C=FLAGS.lr_c, lr_D=FLAGS.lr_d, lr_G=FLAGS.lr_g,
                        lr_E=FLAGS.lr_e, lr_R=FLAGS.lr_g, times=FLAGS.n_times,
                        d_times=FLAGS.d_times, e_times=FLAGS.e_times,
                        imbalanced=FLAGS.imbalanced, batch=FLAGS.batch,
                        epoch=FLAGS.epoch, num=num_loso)

            if not FLAGS.load_mode:
                adan.train_adan()
            else:
                adan.load_adan(FLAGS.model)


        trn_data = sio.loadmat(trn)
        x_train = np.array(trn_data['x'], dtype='float32')
        y_train = np.array(trn_data['y'].ravel(), dtype='int32')

        hx = adan.extract_latent(x_train)

        tst_data = sio.loadmat(tst)
        x_test = np.array(tst_data['x'], dtype='float32')
        y_test = np.array(tst_data['y'].ravel(), dtype='int32')
        htest = adan.extract_latent(x_test)

        folder = os.path.exists(res_path)
        if not folder:
            os.makedirs(res_path)
    
        filelist1 = []
        filelist2 = []
        for n in range(FLAGS.n_times):
            hz, reconz, yz = adan.augment(y_train)
            file1 = syn_path + str(num_loso) + '_' + str(n) + '.h5'
            filelist1.append(file1)
            with h5py.File(file1, 'w') as f1:
                f1.create_dataset('x', data=reconz)
                f1.create_dataset('y', data=yz)
            
            file2 = res_path + '/synencode_trn_loso' + str(num_loso) + '_' + str(n) + '.h5'
            filelist2.append(file2)
            with h5py.File(file2, 'w') as f2:
                f2.create_dataset('x', data=hz)
                f2.create_dataset('y', data=yz)
    
        file3 = res_path + '/encode_trn_loso' + str(num_loso) + '.h5'
        file4 = res_path + '/encode_tst_loso' + str(num_loso) + '.h5'
      
        with h5py.File(file3, 'w') as f3:
            f3.create_dataset('x', data=hx)
            f3.create_dataset('y', data=y_train)
               
        with h5py.File(file4, 'w') as f4:
            f4.create_dataset('x', data=htest)
            f4.create_dataset('y', data=y_test)
        
        sess.close()
        
        syn_combine = syn_path + str(num_loso) + '.h5'
        combine_h5(filelist1, syn_combine)
        syn_enc_comb = res_path + '/synencode_trn_loso' + str(num_loso) + '.h5'
        combine_h5(filelist2, syn_enc_comb)


def plot(path, syn_path, ref_path, num_loso, times):
    path_ = './fig'
    folder = os.path.exists(path_)
    if not folder:
        os.makedirs(path_)

    path_enc = path + '/synencode_trn_loso' + str(num_loso) + '_1.h5'
    tsne_enc = tsnePlot(path_enc, 'h5', 2) 
#    path_enc = path + '/synencode_trn_loso' + str(num_loso) + '.h5'
#    tsne_enc = tsnePlot(path_enc, 'h5', 1+times) 
    output_enc = path_ + '/encode' + str(num_loso) + '.pdf'
    ref_path_enc = path + '/encode_trn_loso' + str(num_loso) + '.h5'
    tsne_enc.plot(ref_path_enc, 'h5', output_enc)

    path_syn = syn_path + str(num_loso) + '_1.h5'
    tsne_syn = tsnePlot(path_syn, 'h5', 2)
    output_syn = path_ + '/syn' + str(num_loso) + '.pdf'
    tsne_syn.plot(ref_path, 'mat', output_syn)

def main(_):
    if FLAGS.dataset == 'iemocap':
        dataset_dir = FLAGS.dataset + '_' + FLAGS.loso + '/' + FLAGS.data_dir
    else:
        dataset_dir = FLAGS.dataset + '_' + FLAGS.loso
        
    syn_path = FLAGS.res_path + '/' + FLAGS.dataset + '_trn_loso'

    for num_loso in range(1, 11):
        trn = '../data/' + dataset_dir + '/' + FLAGS.dataset + '_trn_loso' + str(num_loso) + '.mat'
        tst = '../data/' + dataset_dir + '/' + FLAGS.dataset + '_tst_loso' + str(num_loso) + '.mat'
        augment(FLAGS.res_path, trn, tst, syn_path, num_loso)
        if FLAGS.plot:
            plot(FLAGS.res_path, syn_path, trn, num_loso, FLAGS.n_times)

if __name__ == '__main__':
    tf.app.run()
