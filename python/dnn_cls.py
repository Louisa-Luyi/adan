# Author: M.W. Mak, Dept. of EIE, HKPolyU
# Last update: 4 April. 2020 by Lu YI

from __future__ import print_function
import numpy as np
import time
import os
import scipy.io as sio
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout, Activation
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import sys
from keras import regularizers, optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
import h5py
plt.switch_backend('agg')

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("dataset", "emodb", "dataset to be used, either emodb or iemocap")
tf.app.flags.DEFINE_string("loso", "spk", "leave-one-speaker out or leave-one-session-out (only for iemocap)") #spk or sess
tf.app.flags.DEFINE_string("data_dir", "full", "using full data or improvised only for iemocap")
tf.app.flags.DEFINE_string("res_path", "../result/emodb_adan", "folder to store the generated samples")
tf.app.flags.DEFINE_integer("times", 10, "the ratio: (synthetic samples)/(original samples)")

def main():
    start = time.clock()
    # Define constant
    optimizer = 'adam'    # Can be 'adam', 'sgd', or 'rmsprop'
    activation = 'tanh'   # Can be 'sigmoid', 'tanh', 'softplus', 'softsign', 'relu'
    n_hiddens = [100,100]
    load_mode = False
    n_correct = 0
    n_samples = 0
    class_count = 0
    class_correct = 0
    n_speakers = 10                         # There are 10 speakers in EmoDB
    datadir = FLAGS.res_path

    # Use 1/3 of the GPU memory so that the GPU can be shared by multiple users
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    if FLAGS.dataset == 'iemocap':
        dataset_dir = FLAGS.dataset + '_' + FLAGS.loso + '/' + FLAGS.data_dir
    else:
        dataset_dir = FLAGS.dataset + '_' + FLAGS.loso

#    optimizer = optimizers.Adam(lr=1e-4)

    print('Start cross validation...')
    for i in range(1,n_speakers+1):
        print('Speaker ' + str(i) + ': ', end='')
        
        syn_file = FLAGS.res_path + '/' + FLAGS.dataset + '_trn_loso' + str(i) + '.h5'
        trn_file = '../data/' + dataset_dir + '/' + FLAGS.dataset + '_trn_loso' + str(i) + '.mat'
        tst_file = '../data/' + dataset_dir + '/' + FLAGS.dataset + '_tst_loso' + str(i) + '.mat'

        # Load trn and tst data
        trn_data = sio.loadmat(trn_file)
        tst_data = sio.loadmat(tst_file)

        # Retreive training and test data
        x_train = np.array(trn_data['x'], dtype='float32')
        y_train = np.array(trn_data['y'].ravel(), dtype='int32')
        
        syn_data = h5py.File(syn_file, 'r')
        x_train1 = np.array(syn_data['x'][()])
        y_train1 = np.array(syn_data['y'][()])
        syn_data.close()

        x_train2 = x_train
        y_train2 = y_train
        
        x_val = x_train2
        y_val_ohe = np_utils.to_categorical(y_train2)
        n_emotions = y_val_ohe.shape[1]

        # Select the synthetic samples randomly
        for sel_i in range(n_emotions):
            x_train1_tmp, y_train1_tmp = data_extract(x_train1, y_train1, [sel_i])
            len_tmp = len(x_train1_tmp)
            rand = np.arange(len_tmp)
            np.random.shuffle(rand)
            x_train1_tmp_new = x_train1_tmp[rand]
            y_train1_tmp_new = y_train1_tmp[rand]
            
            l_tmp = int(len_tmp/10)
            x_train = np.concatenate((x_train, x_train1_tmp_new[:int(FLAGS.times*l_tmp)]), axis=0)
            y_train = np.concatenate((y_train, y_train1_tmp_new[:int(FLAGS.times*l_tmp)]), axis=0)
            
        y_train_ohe = np_utils.to_categorical(y_train)

        
        x_test = np.array(tst_data['x'], dtype='float32')
        y_test = np.array(tst_data['y'].ravel(), dtype='int32')

        weight = []
        for wi in range(y_train_ohe.shape[1]):
            weight.append(float(int(len(y_train)/y_train.tolist().count(wi))))

        class_name = np.arange(y_train_ohe.shape[1]).tolist()
        weight_ = dict(zip(class_name, weight))

        # Train DNN
        if not load_mode:
            model = train_dnn(x_train, y_train_ohe, x_val, y_val_ohe, i, FLAGS.times,
                              n_hiddens, optimizer, activation, weight_, load_mode, 
                              datadir, n_epochs=20, bat_size=50)
        else:
            model_name = datadir + '/model/best'+str(FLAGS.times)+'_'+str(i)+'.h5'
            model = load_model(model_name)
        
        # Test DNN
        train_acc, dummy, dummy, dummy, dummy = test_dnn(x_train, y_train, model)
        print('DNN Training accuracy: %.2f%% ' % (train_acc * 100), end='')

        # Get DNN accuracy of current speaker
        test_acc, n_c, n_s, c_c, c_t = test_dnn(x_test, y_test, model)
        n_correct = n_correct + n_c
        n_samples = n_samples + n_s
        class_correct = class_correct + c_c
        class_count = class_count + c_t
        print('DNN Test accuracy: %.2f%% ' % (test_acc * 100))
#        print('Matrix: ', class_correct )
        # Release GPU memory for next fold
        # K.clear_session()

    n_classes = y_train_ohe.shape[1]
    print('Overall DNN accuracy for %d classes with %d features: %.2f%%' %
          (n_classes, (float(n_correct)/float(n_samples) * 100)))

    ratio_arr = class_correct.astype(float)/class_count.astype(float)
    print('Matrix: ', class_correct )
    print('Overall UWA accuracy: %.4f' % np.mean(ratio_arr))
    elapsed = (time.clock()-start)
    print('Time used:', elapsed)


def get_accuracy(pred_labels, true_labels):
    y_pred = np.empty((0, ), dtype='int')
    y_true = np.empty((0, ), dtype='int')
    for i in range(len(pred_labels)):
        y_pred = np.hstack((y_pred, pred_labels[i]))
        y_true = np.hstack((y_true, true_labels[i]))
    n_correct = np.sum(y_true == y_pred, axis=0)
    acc = 100 * n_correct / y_true.shape[0]
    return acc


def train_dnn(x_train, y_train_ohe, x_val, y_val_ohe, num, times, n_hiddens, optimizer, act, weight, load_mode, path, n_epochs=20, bat_size=50):
    np.random.seed(1)

    # Create a DNN
    model = Sequential()

    # Define number of hidden layers and number of nodes in each layer according to n_hiddens
    model.add(Dense(n_hiddens[0], input_dim=x_train.shape[1], activation=act, name='Layer-1'))
    model.add(BatchNormalization(name='L1-BN1'))
    model.add(Dropout(0.5, name='L1-Dropout1'))
    for i in range(1, len(n_hiddens)):
        model.add(Dense(n_hiddens[i], name='Layer-%d' % (i+1), kernel_regularizer=regularizers.l2(0.1)))
        model.add(BatchNormalization(name='L%d-BN%d' % (i+1, i+1)))
        model.add(Activation(act, name='L%d-Act%d' % (i+1, i+1)))
        model.add(Dropout(0.5, name='L%d-Dropout%d' % (i+1, i+1)))
    model.add(Dense(y_train_ohe.shape[1], name='Layer-BeforeSM', kernel_regularizer=regularizers.l2(0.1)))
    model.add(Activation('softmax', name='Layer-AfterSM'))

    # Define loss function and optimizer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # model.summary()

    # Define the callback for early stoping
#    early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=0, verbose=1, mode='auto')

    # Define the callback for best model checking
    model_path = path + '/model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    name = model_path + '/best'+str(times)+'_'+str(num)+'.h5'
    checkpoint = ModelCheckpoint(filepath=name, monitor='val_acc', verbose=0, save_best_only='True', mode='auto')

    # Callback for learning rate decay
    lrdecay = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=0,verbose=0,mode='auto',epsilon=0.0001,cooldown=0,min_lr=0)

    # Perform training class_weight = weight
    history=model.fit(x_train, y_train_ohe, epochs=n_epochs, batch_size=bat_size, verbose=0,
            validation_data=(x_val, y_val_ohe), shuffle=True, callbacks=[checkpoint])

    return model


def test_dnn(X, y, model):
    y_pred = model.predict_classes(X, verbose=0)
    n_correct = np.sum(y == y_pred, axis=0)
    n_samples = X.shape[0]
    test_acc = n_correct / float(n_samples)
    cm = confusion_matrix(y, y_pred)
    class_correct = np.diag(cm)
    class_count = np.sum(cm, axis=1)

    # this loop only for EmoDB
    if len(class_correct) == 6:
        class_correct=np.insert(class_correct,2,0)
        class_count=np.insert(class_count,2,0)

    print(recall_score(y, y_pred, average='macro'))
    print(cm)
    return test_acc, n_correct, n_samples, class_correct, class_count

def data_extract(x, y, num_list):
    x_new = np.array([])
    y_new = np.array([])
    for i in range(len(num_list)):
        index = np.argwhere(y==num_list[i]).ravel()
        if len(x_new) == 0:
            x_new = x[index]
            y_new = y[index]
        else:
            x_new = np.concatenate([x_new, x[index]], axis=0)
            y_new = np.concatenate([y_new, y[index]], axis=0)

    return x_new, y_new


main()

