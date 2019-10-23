import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution1D, Conv2D, MaxPooling2D, MaxPooling1D, LSTM, Embedding
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
    x = Convolution2D(nb_filter, (1, 21), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    return x

def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    feature_list = [x]

    for i in range(nb_layers):
        x = conv_block(x, growth_rate, dropout_rate, weight_decay)
        feature_list.append(x)
        x = Concatenate(axis=concat_axis)(feature_list)
        nb_filter += growth_rate

    return x, nb_filter

def transition_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = Convolution2D(nb_filter, (1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(input)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((1, 2), strides=(1, 2))(x)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

    return x

def createDenseNet(nb_classes, img_dim, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=None,
                     weight_decay=1E-4, verbose=True):

    model_input = Input(shape=img_dim)

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Convolution2D(nb_filter, (1, 21), kernel_initializer="he_uniform", padding="same", name="initial_conv2D", use_bias=False,
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
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

    densenet = Model(inputs=model_input, outputs=x)

    if verbose:
        print("DenseNet-%d-%d created." % (depth, growth_rate))

    return densenet

#define DenseNet parms
ROWS = 1
COLS = 800
CHANNELS = 1
nb_classes = 2
batch_size = 10
nb_epoch = 10000
img_dim = (ROWS,COLS,CHANNELS)
densenet_depth = 103
densenet_growth_rate = 10

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

train_data_normal = load_data('../train_qrs_data/selectedata4s_qrs/N_qrs')
train_data_BBB = load_data('../train_qrs_data/selectedata4s_qrs/BBB_qrs')
train_data_P = load_data('../train_qrs_data/selectedata4s_qrs/P_qrs')
train_data_St = load_data('../train_qrs_data/selectedata4s_qrs/st_qrs')
train_data_Stb = load_data('../train_qrs_data/selectedata4s_qrs/Stb_qrs')
train_data_T = load_data('../train_qrs_data/selectedata4s_qrs/T_qrs')
train_data_V = load_data('../train_qrs_data/selectedata4s_qrs/V_qrs')
train_data_W = load_data('../train_qrs_data/selectedata4s_qrs/W_qrs')
train_data_Other = load_data('../train_qrs_data/selectedata4s_qrs/other_qrs')
train_data_S = load_data('../train_qrs_data/selectedata4s_qrs/S_qrs')
train_data_N_BBB = np.append(train_data_BBB[:148*2], train_data_normal[:1572*2])
train_data_P_St = np.append(train_data_P[:4*2], train_data_St[:290*2])
train_data_Stb_T = np.append(train_data_Stb[:11*2], train_data_T[:282*2])
train_data_V_W = np.append(train_data_V[:471*2], train_data_W[:3*2])
train_data_BBB_N_P_St = np.append(train_data_N_BBB, train_data_P_St)
train_data_Stb_T_V_W = np.append(train_data_Stb_T, train_data_V_W)
train_data_except_other = np.append(train_data_BBB_N_P_St, train_data_Stb_T_V_W)
train_data_Not_S_qrs = np.append(train_data_except_other, train_data_Other[:774*2])
train_data_Whether_S_qrs = np.append(train_data_S, train_data_Not_S_qrs)
Train_Data_Whether_S_qrs = train_data_Whether_S_qrs.reshape(-1, 1, 800)

label_data_S = np.ones(3555*2)
label_data_Not_S = np.zeros(3555*2)
Label_data_train = np.append(label_data_S, label_data_Not_S)

enc = OneHotEncoder()
Label_For_Train = Label_data_train.reshape(-1,1)
enc.fit(Label_For_Train)
Label_for_train =enc.transform(Label_For_Train).toarray()

X_train,X_valid,y_train,y_valid = train_test_split(Train_Data_Whether_S_qrs,Label_for_train,test_size=0.3,random_state=0)
# print(X_train.shape)

checkpoint = keras.callbacks.ModelCheckpoint('DenseNet_Classfication_2_Whether_S_qrs.h5', monitor='val_acc', verbose=1, save_best_only=True)
earlystop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=100, verbose=1, mode='auto')
xtrain = X_train.reshape(-1,ROWS,COLS,CHANNELS)
xvalid = X_valid.reshape(-1,ROWS,COLS,CHANNELS)
model.fit(xtrain, y_train, batch_size=batch_size, epochs=nb_epoch,
         validation_data=(xvalid,y_valid),
         callbacks = [earlystop,checkpoint])