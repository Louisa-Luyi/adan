# Author: M.W. Mak, Dept. of EIE, HKPolyU
# Last update: 4 Apr. 2020 by Lu YI

from __future__ import print_function
import numpy as np
import scipy.io as sio
from sklearn.svm import SVC
from sklearn import linear_model
from keras.utils import np_utils
import h5py
from sklearn.metrics import recall_score
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import sys
import time

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("dataset", "emodb", "dataset to be used, either emodb or iemocap")
tf.app.flags.DEFINE_string("loso", "spk", "leave-one-speaker out or leave-one-session-out (only for iemocap)") #spk or sess
tf.app.flags.DEFINE_string("data_dir", "full", "using full data or improvised only for iemocap")
tf.app.flags.DEFINE_string("res_path", "../result/emodb_adan", "folder to store the generated samples")
tf.app.flags.DEFINE_integer("times", 10, "the ratio: (synthetic samples)/(original samples)")

def main():
    start = time.clock()
    # Define constant
    n_features = 100                       # Totally 382 features for IS09_emotion; 4354 for IS11_speaker_state
    n_speakers = 10                         # There are 10 speakers in EmoDB

    svm_pred = []
    svm_true = []
    print('Start cross validation...')
    print('C=10')
    
    if FLAGS.dataset == 'iemocap':
        dataset_dir = FLAGS.dataset + '_' + FLAGS.loso + '/' + FLAGS.data_dir
    else:
        dataset_dir = FLAGS.dataset + '_' + FLAGS.loso
        
    for i in range(1, n_speakers + 1):

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
        y_train_ohe = np_utils.to_categorical(y_train)
        n_emotions = y_train_ohe.shape[1]
        
        syn_data = h5py.File(syn_file, 'r')
        x_train1 = np.array(syn_data['x'][()])
        y_train1 = np.array(syn_data['y'][()])
        syn_data.close()
        
        # Select the synthetic samples randomly
        for i in range(n_emotions):
            x_train1_tmp, y_train1_tmp = data_extract(x_train1, y_train1, [i])
            len_tmp = len(x_train1_tmp)
            rand = np.arange(len_tmp)
            np.random.shuffle(rand)
            x_train1_tmp_new = x_train1_tmp[rand]
            y_train1_tmp_new = y_train1_tmp[rand]
            
            l_tmp = int(len_tmp/10)
            x_train = np.concatenate((x_train, x_train1_tmp_new[:int(FLAGS.times*l_tmp)]), axis=0)
            y_train = np.concatenate((y_train, y_train1_tmp_new[:int(FLAGS.times*l_tmp)]), axis=0)
        
        x_test = np.array(tst_data['x'], dtype='float32')
        y_test = np.array(tst_data['y'].ravel(), dtype='int32')

        # Train an SVM classifier
        svc = SVC(C=10,gamma='auto', kernel='rbf')#,decision_function_shape='ovo')#, class_weight='balanced')
#        svc = linear_model.SGDClassifier()
        svc.fit(x_train, y_train)

        # Test the SVM classifier
        pred = svc.predict(x_test)
        cm = confusion_matrix(y_test, pred)
        print(cm)
        acc, uar = get_accuracy([pred], [y_test])
        print('SVM accuracy = %.2f%%' % acc)
        print('SVM UAR = %.4f' % uar)
        svm_pred.append(svc.predict(x_test))
        svm_true.append(y_test)

    n_classes = np.max(y_train) + 1
    acc_all, uar_all = get_accuracy(svm_pred, svm_true)
    print('Overall SVM accuracy for %d classes with %d features/class: %.2f%%' %
          (n_classes, n_features, acc_all))
    print('Overall SVM UWA: %.4f' % uar_all)

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
    uar = recall_score(y_true, y_pred, average='macro')
    return acc, uar

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


