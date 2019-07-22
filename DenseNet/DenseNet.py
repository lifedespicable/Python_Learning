import os
import gc
import sys
import csv
import copy
import keras
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from keras.utils.np_utils import to_categorical
from keras.layers import Input, Flatten, Dense, Activation, Dropout, BatchNormalization, \
                         Conv1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, \
                         GlobalMaxPooling1D, concatenate, add, Lambda

from keras.models import Model, Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.optimizers import SGD, Adam, RMSprop
from keras import backend as K

# from denModel import conv_block, dense_block, transition_block

pd.set_option('display.height',1000)
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)

CLASSES = 9
lead_num = 12
seq_length = 7500
paint = 1


def mat_to_ecg(mat_path):
    data = sio.loadmat(mat_path)['ECG']['data'][0][0]
    return data


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_mat_path, labels, batch_size=1, n_channels=12,
                 n_classes=9, shuffle=True):
        self.batch_size = batch_size
        self.labels = labels
        self.list_mat_path = list_mat_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_mat_path) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_mat_path = [self.list_mat_path[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_mat_path, indexes)
        return X, y

    def on_epoch_end(self):
        """
        Update indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_mat_path))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_mat_path_temp, indexes):
        """
        Generates data containing batch size samples
        eg: X.shape = [n_samples, seq_length, n_channels]
        """
        # Initialization
        X = np.empty((self.batch_size, 7500, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_mat_path_temp):
            # Store sample
            mat_path = list_mat_path_temp[i]
            ecg_data = mat_to_ecg(mat_path)
            ecg_data = ecg_data.T
            if self.batch_size == 1:
                X = np.reshape(ecg_data, (self.batch_size, ecg_data.shape[0], ecg_data.shape[1]))
            else:
                X[i] = ecg_data[:7500]
            y[i] = self.labels[indexes[i]]
        return X, y


def acc(a, b):
    return keras.metrics.sparse_categorical_accuracy(a, b)


class F1(keras.layers.Layer):
    def __init__(self, classes, name='F1', **kwargs):
        """
        classes: the output_dims
        """
        super(F1, self).__init__(name=name, **kwargs)
        self.classes = classes
        self.stateful = True
        self.zeros = np.zeros((classes, classes), dtype=np.float32)
        # something like confusion matrix
        self.confusion = K.variable(value=self.zeros, dtype='float32')

    pass

    def reset_states(self):
        """
        when the epoch ends, clear the confusion matrix
        """
        K.set_value(self.confusion, self.zeros)

    def __call__(self, y_true, y_pred):
        """
        when run the instant,call this function to calculate f1_score
        """
        true = K.one_hot(K.cast(K.max(y_true, axis=-1), dtype=np.uint8), self.classes)
        pred = K.one_hot(K.argmax(y_pred, axis=-1), self.classes)
        # calculate true.T * pred
        batch = K.dot(K.transpose(true), pred)
        self.add_update(K.update_add(self.confusion, batch), inputs=[y_true, y_pred])
        cur = self.confusion + batch
        # sum alongside column
        d1 = K.sum(cur, axis=0)
        # sum alongside row
        d2 = K.sum(cur, axis=1)
        # return the tensor's diagonal(对角线)
        diag = Lambda(tf.diag_part)(cur)
        return 2. * K.mean(diag / (d1 + d2 + 0.00001))


class AccF1MetricCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_acc = []
        self.val_acc = []
        self.f1 = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.f1.append(logs.get('val_F1'))

    #         val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
    #         val_pred = np.zeros((val_predict.shape[0],1))
    #         val_target = self.validation_data[1]
    #         for i in range(val_predict.shape[0]):
    #             val_pred[i,:] = np.argmax(val_predict[i,:])
    #         _val_f1 = f1_score(val_target, val_predict)
    #         self.f1.append(_val_f1)
    #         f1_score = F1(CLASSES)
    #         self.f1.append(f1_score(self.validation_data[1], self.model.predict(self.validation_data[0])))

    def plot_metric(self, mode):
        iterations = range(len(self.train_acc))
        plt.figure(figsize=(8, 4))
        # acc
        plt.plot(iterations, self.train_acc, 'r', label='train_acc')
        # val_acc
        plt.plot(iterations, self.val_acc, 'g', label='val_acc')
        # f1
        plt.plot(iterations, self.f1, 'b', label='f1_score')
        plt.grid(True)
        ax = plt.axes()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Acc-F1')
        plt.legend(loc='lower right')
        plt.ylim((0.5, 1))
        plt.show()
        return self.train_acc, self.val_acc, self.f1


def plot_Average(*args):
    color = sns.color_palette()
    plt.figure(figsize=(8, 4))
    plt.grid(True)
    ax = plt.axes()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric')
    for cnt, item in enumerate(args):
        data = np.zeros((6, len(item[0])))
        print (data.shape)
        for i in range(len(item)):
            #             print data[i], item[i]
            data[i] = item[i]
        item_mean = np.mean(item[0:5], axis=0)
        length = range(item.shape[1])
        plt.plot(length, item_mean, color=color[cnt], label='Metric_' + str(cnt + 1))
    plt.legend(loc='lower right')
    plt.ylim((0.5, 0.8))
    plt.show()


def get_mat_label_list():
    ref_dict = {}
    with open('REFERENCE.csv', "r")as reference:
        csv_reader = csv.reader(reference)
        for index, ref in enumerate(csv_reader):
            if index == 0:
                continue
            value = ref[1:]
            try:
                value.remove("")
                value.remove("")
            except:
                pass
            ref_dict[ref[0]] = int(value[0]) - 1
    return ref_dict

def get_mat_file_path():
    mat_path_dict = {}
    for mats in next(os.walk('Data')):
        for mat in mats:
            if mat.endswith(".mat"):
                mat_path_dict[mat.split('.')[0]] = 'Data/'+mat
    return mat_path_dict

def get_txt_list(train_txt_path, val_txt_path):
    with open(train_txt_path, 'r') as txt:
        train_txt_list = txt.readlines()
        train_txt_list = list(train_txt_list)
        #取前5位 eg A0001\n 取A0001
    train_txt_list = [x[:5] for x in train_txt_list]
    with open(val_txt_path, 'r') as txt:
        test_txt_list = txt.readlines()
        test_txt_list = list(test_txt_list)
        test_txt_list = [x[:5] for x in test_txt_list]
    return train_txt_list, test_txt_list

def __conv_block(ip, nb_filter, kernel_size=15, dropout_rate=None):
    x = BatchNormalization(epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    inter_channel = nb_filter * 4
    x = Conv1D(inter_channel, 1, padding='same')(x)
    x = BatchNormalization(epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    x = Conv1D(nb_filter, kernel_size, strides=2, padding='same')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x

def __dense_block(x, nb_layers, nb_filter, growth_rate, kernel_size, dropout_rate=None, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        dropout_rate: dropout rate
        grow_nb_filters: flag to decide to allow number of filters to grow
    Returns: keras tensor with nb_layers of conv_block appended
    '''
    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, kernel_size, dropout_rate)
        x = MaxPooling1D(padding="same")(x)
        x = concatenate([x, cb])
        if grow_nb_filters:
            nb_filter += growth_rate
    return x, nb_filter

def __transition_block(ip, nb_filter, dropout_rate=None, compression=1.0):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''
    x = BatchNormalization(epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    x = Conv1D(int(nb_filter * compression), 1, padding='same')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling1D(strides=2)(x)

    return x

def dense_net_modified():
    nb_layers = 4
    nb_filter = 32
    growth_rate = 32
    kernel_size = 15
    compression = 0.4
    X = Input(shape=[None, 12])
    net = X
    net = Conv1D(nb_filter, kernel_size, strides=2, activation='relu')(net)
    net = MaxPooling1D(3, strides=2)(net)
    net, nb_filter = __dense_block(net, nb_layers, nb_filter, growth_rate, kernel_size)
    nb_filter = int(nb_filter * compression)
    net = __transition_block(net, nb_filter)
    net, nb_filter = __dense_block(net, nb_layers, nb_filter, growth_rate, kernel_size)
    net = GlobalAveragePooling1D()(net)
    net = Dense(CLASSES ** 2, activation='relu')(net)
    net = Dense(CLASSES, activation=tf.nn.softmax)(net)
    prob = net
    model = Model(inputs=[X], outputs=[prob])
    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=[acc, F1(CLASSES)])
    return model
