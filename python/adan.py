
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 15:56:11 2019

Network structure for ADAN

@author: Lu YI
@organisation: Dept. of EIE, The Hong Kong Polytechnic University
"""

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import np_utils
from tqdm import tqdm
import scipy.io as sio
from utils import leaky_relu, norm, show_result, show_result2

plt.switch_backend('agg')

class ADAN:
    def __init__(self, trn_dir, sess, latent_dim=100, z_dim=2, lr_C=1e-4, lr_D=1e-5, 
                 lr_G=1e-4, lr_E=1e-4, lr_R=1e-3, times=1, d_times=1, e_times=1,
                 imbalanced=False, batch=64, epoch=10, num=0):

        self.trn_dir = trn_dir
        self.sess = sess
        self.latent = latent_dim # Dimension of the latent vector (output of the encoder and the generator).
        self.zdim = z_dim # Dimension of the random vector (input of the generator).
        self.lr_C = lr_C
        self.lr_D = lr_D
        self.lr_G = lr_G
        self.lr_E = lr_E
        self.lr_R = lr_R
        self.times = times # The ratio: (synthetic samples)/(original samples).
        self.d_times = d_times # The number of training times for the discriminator in each iteration.
        self.e_times = e_times # The number of training times for the encoder in each iteration.
        self.imbalanced = imbalanced
        self.batch = batch
        self.epoch = epoch
        self.num = num # The number of the fold for cross-validation.
        
        
    # Encoder
    def encoder(self, x, out_dim):
        with tf.variable_scope('encoder'):
            w_init = tf.contrib.layers.xavier_initializer()

            dense1 = tf.layers.dense(x, 800, kernel_initializer = w_init)
            relu1 = leaky_relu(dense1, 0.2)
            relu1 = tf.layers.batch_normalization(relu1)
#            relu1 = tf.nn.tanh(dense1)
#            relu1 = tf.nn.dropout(relu1,0.5)

            dense2 = tf.layers.dense(relu1, 800, kernel_initializer = w_init)
            relu2 = leaky_relu(dense2, 0.2)
            relu2 = tf.layers.batch_normalization(relu2)
#            relu2 = tf.nn.tanh(dense2)
#            relu2 = tf.nn.dropout(relu2,0.5)
            
            output_layer = tf.layers.dense(relu2, out_dim, kernel_initializer = w_init, name='encode_layer')
            output_layer = tf.nn.tanh(output_layer)
            
            return output_layer
        
        
    
    # Generator
    def generator(self, z, y, out_dim, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            w_init = tf.contrib.layers.xavier_initializer()
            cat1 = tf.concat([z, y], 1)
        
            dense1 = tf.layers.dense(cat1, 100, kernel_initializer = w_init)
            relu1 = leaky_relu(dense1, 0.2)
#            relu1 = tf.nn.tanh(dense1)
#            relu1 = tf.nn.dropout(relu1,rate)

            dense2 = tf.layers.dense(relu1, 100, kernel_initializer = w_init)
            relu2 = leaky_relu(dense2, 0.2)
#            relu2 = tf.nn.tanh(dense2)
#            relu2 = tf.nn.dropout(relu2,rate)
        
            output_layer = tf.layers.dense(relu2, out_dim, kernel_initializer = w_init)
            output_layer = tf.nn.tanh(output_layer)
            
            return output_layer
    
    
    # Discriminator
    def discriminator(self, x, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            w_init = tf.contrib.layers.xavier_initializer()

            dense1 = tf.layers.dense(x, 100, kernel_initializer = w_init)
            relu1 = leaky_relu(dense1, 0.2)
            #relu1 = tf.nn.tanh(dense1)

            dense2 = tf.layers.dense(relu1, 100, kernel_initializer = w_init)
            relu2 = leaky_relu(dense2, 0.2)
            #relu2 = tf.nn.tanh(dense2)
        
            vadility = tf.layers.dense(relu2, 1, kernel_initializer = w_init)
            prob = tf.nn.sigmoid(vadility)
            
            return vadility, prob
        
    # Decoder
    def decoder(self, h, fdim, reuse=False):
        with tf.variable_scope('decoder', reuse=reuse):
            w_init = tf.contrib.layers.xavier_initializer()

            dense1 = tf.layers.dense(h, 800, kernel_initializer = w_init)
            relu1 = leaky_relu(dense1, 0.2)
            relu1 = tf.layers.batch_normalization(relu1)
#            relu1 = tf.nn.tanh(dense1)
#            relu1 = tf.nn.dropout(relu1,0.5)

            dense2 = tf.layers.dense(relu1, 800, kernel_initializer = w_init)
            relu2 = leaky_relu(dense2, 0.2)
            relu2 = tf.layers.batch_normalization(relu2)
#            relu2 = tf.nn.tanh(dense2)
#            relu2 = tf.nn.dropout(relu2,0.5)
        
            output_layer = tf.layers.dense(relu2, fdim, kernel_initializer = w_init)
            output_layer = tf.nn.tanh(output_layer)
    
            return output_layer
    
    
    # Classification
    def classification(self, x, y_dim, reuse=False):
        with tf.variable_scope('classifier', reuse=reuse):
            w_init = tf.contrib.layers.xavier_initializer()

            dense1 = tf.layers.dense(x, 400, kernel_initializer = w_init)
            relu1 = leaky_relu(dense1, 0.2)
#            relu1 = tf.nn.tanh(dense1)
#            relu1 = tf.nn.dropout(relu1,0.5)

            dense2 = tf.layers.dense(relu1, 400, kernel_initializer = w_init)
            relu2 = leaky_relu(dense2, 0.2)
#            relu2 = tf.nn.tanh(dense2)
#            relu2 = tf.nn.dropout(relu2,0.5)
        
            y_label = tf.layers.dense(relu2, y_dim, kernel_initializer = w_init)
            y_ = tf.nn.softmax(y_label, name='emotion_layer')
            
            return y_label, y_
        
    
    # Load data from .mat file
    def load_data(self):
        trn_data = sio.loadmat(self.trn_dir)  
        x_train = np.array(trn_data['x'], dtype='float32')
        self.x_train, self.max_ = norm(x_train)
        self.y_train = np.array(trn_data['y'].ravel(), dtype='int32')
        self.y_train_ohe = np_utils.to_categorical(self.y_train)
        self.fdim = self.x_train.shape[1]
        self.numClasses = self.y_train_ohe.shape[1]
        
        weight=[]
        if self.imbalanced:
            numS = []
            for ll in range(self.numClasses):
                numS.append(float(int(self.y_train.tolist().count(ll))))
            
            for l in range(self.numClasses):
                weight.append(float(int(max(numS)/self.y_train.tolist().count(l))))
        else:
            for l in range(self.numClasses):
                weight.append(1.0)
        
        self.class_weights = tf.constant(weight)
    
    # Build the model. 
    # If the dataset is imbalanced, set the imbalanced to True to use the 
    # weight loss strategy.
    def build_adan(self):
        # Input
        self.load_data()
        self.x = tf.placeholder(tf.float32, shape=(None, self.fdim))
        self.y = tf.placeholder(tf.float32, shape=(None, self.numClasses))
        self.z = tf.placeholder(tf.float32, shape=(None, self.zdim))
        self.g_z = tf.placeholder(tf.float32, shape=(None, self.latent))
#        self.dropout = tf.placeholder(tf.float32)
        
        # Network
        self.Gz = self.generator(self.z, self.y, self.latent, reuse=False)
        self.h = self.encoder(self.x, self.latent)
        self.real_logit, self.prob_real = self.discriminator(self.h, reuse=False)
        self.fake_logit, self.prob_fake = self.discriminator(self.Gz, reuse=True)
        self.real_label, self.real_softmax = self.classification(self.h, self.numClasses,reuse=False)
        self.fake_label, self.fake_softmax = self.classification(self.Gz, self.numClasses, reuse=True)
        self.recon_x = self.decoder(self.h, self.fdim, reuse=False)
        self.recon_z = self.decoder(self.g_z, self.fdim, reuse = True)
       
        self.D_loss_r = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logit, 
                                                        labels=tf.ones_like(self.real_logit)))
        self.D_loss_f = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit, 
                                                        labels=tf.zeros_like(self.fake_logit)))
        self.D_loss = 0.5*self.D_loss_r + 0.5*self.D_loss_f
        
        self.weights = tf.reduce_sum(self.class_weights * self.y, axis=1)
        self.Cg_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.fake_label,labels=self.y) * self.weights)

        self.G_loss = -self.D_loss_f + 1.0*self.Cg_loss
    
        self.C_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.real_label,labels=self.y) * self.weights)
            
        self.R_loss = tf.reduce_mean(tf.squared_difference(self.x, self.recon_x))
        
        self.E_loss = self.R_loss + self.C_loss 
        
        self.saver = tf.train.Saver(max_to_keep=10)
        
        
    # Train the model
    def train_adan(self):
        
        stop = False
        pre_epoch = 0
        count = 0
        
        self.build_adan()
    
        # Trainable variables
        T_vars = tf.trainable_variables()
        self.D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
        self.C_vars = [var for var in T_vars if var.name.startswith('classifier')]
        self.G_vars = [var for var in T_vars if var.name.startswith('generator')]
        self.E_vars = [var for var in T_vars if var.name.startswith('encoder')]
        self.R_vars = [var for var in T_vars if var.name.startswith('decoder')]
    
        # Optimizer
        with tf.control_dependencies(tf.get_collection_ref(tf.GraphKeys.UPDATE_OPS)):
            D_optimizer = tf.train.AdamOptimizer(self.lr_D, beta1=0.5, epsilon=1e-5).minimize(self.D_loss, var_list=self.D_vars)
            C_optimizer = tf.train.AdamOptimizer(self.lr_C, beta1=0.5, epsilon=1e-5).minimize(self.C_loss, var_list=self.C_vars)
            G_optimizer = tf.train.AdamOptimizer(self.lr_G, beta1=0.5, epsilon=1e-5).minimize(self.G_loss, var_list=self.G_vars)
            E_optimizer = tf.train.AdamOptimizer(self.lr_E, beta1=0.5, epsilon=1e-5).minimize(self.E_loss, var_list=self.E_vars)
            R_optimizer = tf.train.AdamOptimizer(self.lr_R, beta1=0.5, epsilon=1e-5).minimize(self.R_loss, var_list=self.R_vars)
    
        tf.global_variables_initializer().run()
        train_op = tf.group(C_optimizer, R_optimizer, E_optimizer, G_optimizer)

        train_hist = {'D_losses':[], 'C_losses':[], 'G_losses':[], 'R_losses':[]}
        acc_hist = {'f_acc':[], 'r_acc':[], 'f_std':[], 'r_std':[]}
    
        with tqdm(total=self.epoch) as pbar:
            for e in range(self.epoch):
                if stop != True:
                    pbar.update(1)
                    index = np.arange(len(self.x_train))
                    np.random.shuffle(index)
                    x_train = self.x_train[index]
                    y_train_ohe = self.y_train_ohe[index]
                    for iter in range(x_train.shape[0] // self.batch):
                        x_ = x_train[iter * self.batch:(1+iter)*self.batch]
                        y_ = y_train_ohe[iter * self.batch:(1+iter)*self.batch]
                        z_ = np.random.normal(0, 1, (self.batch, self.zdim))
                        
                        # Training the discriminator and then freezing its weights
                        for d in range(self.d_times):
                            self.sess.run(D_optimizer, {self.x: x_, self.y: y_, self.z: z_})
                        loss_d = self.sess.run(self.D_loss, {self.x: x_, self.y: y_, self.z: z_})
                        
                        self.sess.run(train_op, {self.x: x_, self.y: y_, self.z: z_})
                        loss_c, loss_r, loss_e, loss_g = self.sess.run([self.C_loss, self.R_loss, self.E_loss, self.G_loss], 
                                                                       {self.x: x_, self.y: y_, self.z: z_})
              
                    train_hist['D_losses'].append(loss_d)
                    train_hist['C_losses'].append(loss_c)
                    train_hist['G_losses'].append(loss_g) 
                    train_hist['R_losses'].append(loss_r)


                    z_trn = np.random.normal(0, 1, (self.x_train.shape[0], self.zdim))
                    prob_f = self.sess.run(self.prob_fake, {self.z: z_trn, self.y: self.y_train_ohe})
                    prob_f_mean = np.mean(prob_f)

                    prob_trn = self.sess.run(self.prob_real, {self.x: self.x_train, self.y: self.y_train_ohe})
                    prob_trn_mean = np.mean(prob_trn)

                    acc_hist['f_acc'].append(prob_f_mean)
                    acc_hist['r_acc'].append(prob_trn_mean)
                    acc_hist['f_std'].append(np.std(prob_f))
                    acc_hist['r_std'].append(np.std(prob_trn))
 
                    prob_r = self.sess.run(self.prob_real, {self.x: self.x_train})
                    prob_mean = np.mean(prob_r)
                    
                    # Early stop
                    if e>200 and np.abs(prob_mean-0.5) <= 0.01 and np.abs(prob_trn_mean-0.5) <= 0.01 and np.abs(prob_f_mean-0.5) <= 0.01:
                        if np.abs(e-pre_epoch)<=2 and count == 3:
                            stop = True
                            show_result(train_hist, e, self.num)
                            show_result2(acc_hist,e, self.num)
                            model_name = './model_adan/model'+str(e)+'.ckpt'
                            self.saver.save(self.sess, model_name)
                        elif np.abs(e-pre_epoch) <= 2:
                            pre_epoch = e
                            count = count + 1
                        else:
                            pre_epoch = e
                
                    if e == self.epoch-1:
                        show_result(train_hist, e, self.num)
                        show_result2(acc_hist, e, self.num)
                        model_name = './model_adan/model_loso' + str(self.num) + '_final.ckpt'
                        self.saver.save(self.sess, model_name)
                        
    def build_wadan(self):
        # Input
        self.load_data()
        self.x = tf.placeholder(tf.float32, shape=(None, self.fdim))
        self.y = tf.placeholder(tf.float32, shape=(None, self.numClasses))
        self.z = tf.placeholder(tf.float32, shape=(None, self.zdim))
        self.g_z = tf.placeholder(tf.float32, shape=(None, self.latent))
#        self.dropout = tf.placeholder(tf.float32)
        
        # Network
        self.Gz = self.generator(self.z, self.y, self.latent, reuse=False)
        self.h = self.encoder(self.x, self.latent)
        self.real_logit, self.prob_real = self.discriminator(self.h, reuse=False)
        self.fake_logit, self.prob_fake = self.discriminator(self.Gz, reuse=True)
        self.real_label, self.real_softmax = self.classification(self.h, self.numClasses,reuse=False)
        self.fake_label, self.fake_softmax = self.classification(self.Gz, self.numClasses, reuse=True)
        self.recon_x = self.decoder(self.h, self.fdim, reuse=False)
        self.recon_z = self.decoder(self.g_z, self.fdim, reuse = True)
        
        self.D_loss = tf.reduce_mean(self.real_logit) - tf.reduce_mean(self.fake_logit)

        # gradient penalty
        epsilon = np.random.uniform(0., 1., (self.batch, 1))
        interpolated = self.h + epsilon*(self.Gz-self.h)
        inte_logit, _ = self.discriminator(interpolated, reuse=True)
        gradients = tf.gradients(inte_logit, [interpolated])[0]
        grad_l2 = tf.pow(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]), 5)
        gradient_penalty = tf.reduce_mean(grad_l2)

        self.D_loss = self.D_loss + 10.0 * gradient_penalty

        self.weights = tf.reduce_sum(self.class_weights * self.y, axis=1)
        self.Cg_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.fake_label,labels=self.y) * self.weights)

        self.G_loss = -self.D_loss + 0.1*self.Cg_loss
    
        self.C_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.real_label,labels=self.y) * self.weights)
            
        self.R_loss = tf.reduce_mean(tf.squared_difference(self.x, self.recon_x))
        
        self.E_loss = self.C_loss + self.R_loss 
        
        self.saver = tf.train.Saver(max_to_keep=10)
        
    def train_wadan(self):
        
        self.build_wadan()
        
        # Trainable variables
        T_vars = tf.trainable_variables()
        self.D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
        self.C_vars = [var for var in T_vars if var.name.startswith('classifier')]
        self.G_vars = [var for var in T_vars if var.name.startswith('generator')]
        self.E_vars = [var for var in T_vars if var.name.startswith('encoder')]
        self.R_vars = [var for var in T_vars if var.name.startswith('decoder')]
        
        # Optimizer
        with tf.control_dependencies(tf.get_collection_ref(tf.GraphKeys.UPDATE_OPS)):
            D_optimizer = tf.train.AdamOptimizer(self.lr_D, beta1=0.5, epsilon=1e-5).minimize(self.D_loss, var_list=self.D_vars)
            C_optimizer = tf.train.AdamOptimizer(self.lr_C, beta1=0.5, epsilon=1e-5).minimize(self.C_loss, var_list=self.C_vars)
            G_optimizer = tf.train.AdamOptimizer(self.lr_G, beta1=0.5, epsilon=1e-5).minimize(self.G_loss, var_list=self.G_vars)
            E_optimizer = tf.train.AdamOptimizer(self.lr_E, beta1=0.5, epsilon=1e-5).minimize(self.E_loss, var_list=self.E_vars)
            R_optimizer = tf.train.AdamOptimizer(self.lr_R, beta1=0.5, epsilon=1e-5).minimize(self.R_loss, var_list=self.R_vars)

        tf.global_variables_initializer().run()
        train_op = tf.group(R_optimizer, E_optimizer)
        
        train_hist = {'D_losses':[], 'C_losses':[], 'G_losses':[], 'R_losses':[]}
    
        with tqdm(total=self.epoch) as pbar:
            for e in range(self.epoch):
#                start = time.clock()
                pbar.update(1)
                index = np.arange(len(self.x_train))
                np.random.shuffle(index)
                x_train = self.x_train[index]
                y_train_ohe = self.y_train_ohe[index]
                for iter in range(x_train.shape[0] // self.batch):
#                        start = time.clock()
                    x_ = x_train[iter * self.batch:(1+iter)*self.batch]
                    y_ = y_train_ohe[iter * self.batch:(1+iter)*self.batch]
                    z_ = np.random.normal(0, 1, (self.batch, self.zdim))

                    self.sess.run(C_optimizer, {self.x: x_, self.y: y_, self.z: z_})
                    loss_c = self.sess.run(self.C_loss, {self.x: x_, self.y: y_, self.z: z_})
                    
                    for d in range(self.d_times):
                        self.sess.run(D_optimizer, {self.x: x_, self.y: y_, self.z: z_})
                    loss_d = self.sess.run(self.D_loss, {self.x: x_, self.y: y_, self.z: z_})
                     
                    for e_t in range(self.e_times):
                        self.sess.run(train_op, {self.x: x_, self.y: y_, self.z: z_})
                    loss_r = self.sess.run(self.R_loss, {self.x: x_, self.y: y_, self.z: z_})
                    
                    self.sess.run(G_optimizer, {self.x: x_, self.y: y_, self.z: z_})
                    loss_g = self.sess.run(self.G_loss, {self.x: x_, self.y: y_, self.z: z_})
                        
 #                       elapse = (time.clock()-start)
 #                       print('Time used:', elapse)
                
                train_hist['D_losses'].append(loss_d)
                train_hist['C_losses'].append(loss_c)
                train_hist['G_losses'].append(loss_g) 
                train_hist['R_losses'].append(loss_r)
                
                if e == self.epoch-1:
                    show_result(train_hist, e, self.num)
                    model_name = './model_wadan/model_loso' + str(self.num) + '_final.ckpt'
                    self.saver.save(self.sess, model_name)
                        
    # Generating process after running                      
    def augment(self, y_data):
        y_ohe = np_utils.to_categorical(y_data)
        
        z_sample = np.random.normal(0, 1, (y_data.shape[0], self.zdim))
        G = self.sess.run(self.Gz, {self.z:z_sample, self.y: y_ohe})
        reconz = self.sess.run(self.recon_z, {self.g_z:G, self.y:y_ohe})
        y_f = y_data
        
        reconz = reconz*self.max_
        
        return G, reconz, y_f
    
    # Extract latent vectors from the output of the encoder.
    def extract_latent(self, x_data):
        h = self.sess.run(self.h, {self.x: x_data})
        
        return h

    def load_adan(self, ckpt_path):
        self.build_adan()
        self.saver.restore(self.sess, ckpt_path)
        print("Success to read {}".format(ckpt_path))
            
    def load_wadan(self, ckpt_path):
        self.build_wadan()
        self.saver.restore(self.sess, ckpt_path)
        print("Success to read {}".format(ckpt_path))
