import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution1D, Conv2D, MaxPooling1D, MaxPooling1D, LSTM, Embedding
from keras.optimizers import SGD
import keras.backend as K
import random
import re
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.utils import np_utils
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback, LearningRateScheduler
from keras.regularizers import l2
from keras.layers import *
from keras.models import *
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import OneHotEncoder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 使用第二块GPU（从0开始）

def conv_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
    x = Activation('relu')(input)
    x = Convolution1D(nb_filter, (21), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    return x

def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    concat_axis = 1 if K.image_data_format() == "th" else -1

    feature_list = [x]

    for i in range(nb_layers):
        x = conv_block(x, growth_rate, dropout_rate, weight_decay)
        feature_list.append(x)
        x = Concatenate(axis=concat_axis)(feature_list)
        nb_filter += growth_rate

    return x, nb_filter

def transition_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
    concat_axis = 1 if K.image_data_format() == "th" else -1

    x = Convolution1D(nb_filter, (1), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(input)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling1D(2)(x)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

    return x

def createDenseNet(nb_classes, img_dim, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=None,
                     weight_decay=1E-4, verbose=True):

    model_input = Input(shape=img_dim)

    concat_axis = 1 if K.image_data_format() == "th" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Convolution1D(nb_filter, (21), kernel_initializer="he_uniform", padding="same", name="initial_conv1D", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(model_input)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                            beta_regularizer=l2(weight_decay))(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)
        # add transition_block
        x = transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # The last dense_block does not have a transition_block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = Activation('relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

    densenet = Model(inputs=model_input, outputs=x)

    if verbose:
        print("DenseNet-%d-%d created." % (depth, growth_rate))

    return densenet

#define DenseNet parms
COLS = 6000
CHANNELS = 1
nb_classes = 2
batch_size = 10
nb_epoch = 10000
img_dim = (COLS,CHANNELS)
densenet_depth = 16
densenet_growth_rate = 2

model = createDenseNet(nb_classes=nb_classes,img_dim=img_dim,depth=densenet_depth,
                  growth_rate = densenet_growth_rate)

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

def load_data(path):
    # read data from data file
    temp_data = []
    namelist = [x for x in os.listdir(path)]
    for i in range( len(namelist) ):
        if namelist[i].endswith('npy'):
            temp_data_one = np.load(path + '/' + namelist[i])
            temp_data.append(temp_data_one[0])
            temp_data.append(temp_data_one[1])
    train_data = np.array(temp_data)
    return train_data

train_data_normal = load_data('../train_data/N')
train_data_AF = load_data('../train_data/AF')
train_data_Arr = load_data('../train_data/Arr')
train_data_Ab = load_data('../train_data/Ab')
train_data_other = load_data('../train_data/other')
true_data_Ab = train_data_Ab[:3*2]
test_data_Ab = train_data_Ab[3*2:4*2]
true_data_AF = train_data_AF[:229*2]
test_data_AF = train_data_AF[229*2:]
true_data_Arr = train_data_Arr[:25*2]
test_data_Arr = train_data_Arr[25*2 : 31*2]
true_data_N  = train_data_normal[:60*2]
test_data_N  = train_data_normal[60*2 : 75*2]
true_data_other = train_data_other[:141*2]
test_data_other = train_data_other[141*2: 176*2]
true_data_Ab_Arr = np.append(true_data_Ab, true_data_Arr)
true_data_N_Other = np.append(true_data_N, true_data_other)
true_data_Not_AF = np.append(true_data_Ab_Arr, true_data_N_Other)
true_data = np.append(true_data_AF, true_data_Not_AF)
test_data_Ab_Arr = np.append(test_data_Ab, test_data_Arr)
test_data_N_Other = np.append(test_data_N, test_data_other)
test_data_Not_AF = np.append(test_data_Ab_Arr, test_data_N_Other)
test_data_all = np.append(test_data_AF, test_data_Not_AF)
Train_Data_all = true_data.reshape(-1, 6000)
Test_Data_all = test_data_all.reshape(-1, 6000)

label_train_data_AF = np.ones(458)
label_train_data_Not_AF = np.zeros(458)
Label_data_train = np.append(label_train_data_AF, label_train_data_Not_AF)
Label_test_AF = np.ones(114)
Label_test_Not_AF = np.zeros(114)
Label_data_test = np.append(Label_test_AF, Label_test_Not_AF)

enc = OneHotEncoder()
Label_For_Train = Label_data_train.reshape(-1,1)
enc.fit(Label_For_Train)
Label_for_train =enc.transform(Label_For_Train).toarray()

X_train,X_valid,y_train,y_valid = train_test_split(Train_Data_all,Label_for_train,test_size=0.25,random_state=0)
# print(X_train.shape)

checkpoint = keras.callbacks.ModelCheckpoint('DenseNet_Classfication_2_Whether_AF_dimension_of_1_subject_oriented.h5', monitor='val_acc', verbose=1, save_best_only=True)
earlystop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=100, verbose=1, mode='auto')
xtrain = X_train.reshape(-1,COLS,CHANNELS)
xvalid = X_valid.reshape(-1,COLS,CHANNELS)
model.fit(xtrain, y_train, batch_size=batch_size, epochs=nb_epoch,
         validation_data=(xvalid,y_valid),
         callbacks = [earlystop,checkpoint])