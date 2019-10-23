import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution1D, Conv1D, MaxPooling1D, MaxPooling1D, LSTM, Embedding
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 使用第二块GPU（从0开始）

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

def createDenseNet(nb_classes, img_dim, depth=39, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=None,
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
COLS = 800
CHANNELS = 1
nb_classes = 2
batch_size = 6
nb_epoch = 10000
img_dim = (COLS,CHANNELS)
densenet_depth = 40
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

train_data_normal = load_data('../train_qrs_data/QRS_data/N_qrs')
train_data_T = load_data('../train_qrs_data/QRS_data/T_qrs')
train_data_St = load_data('../train_qrs_data/QRS_data/st_qrs')
train_data_SN = load_data('../train_qrs_data/QRS_data/S_qrs_')
train_data_VN = load_data('../train_qrs_data/QRS_data/V_qrs_')
train_data_WN = load_data('../train_qrs_data/QRS_data/W_qrs_')
train_data_V = load_data('../train_qrs_data/QRS_data/V_qrs')
train_data_W = load_data('../train_qrs_data/QRS_data/W_qrs')
train_data_S = load_data('../train_qrs_data/QRS_data/S_qrs')
train_data_BBB = load_data('../train_qrs_data/QRS_data/BBB_qrs')
train_N_T = np.append(train_data_normal[:202*2], train_data_T[:35*2])
train_St_SN = np.append(train_data_St[:37*2],train_data_SN[:191*2])
train_N_except_VN =  np.append(train_N_T, train_St_SN)
train_data_N = np.append(train_N_except_VN, train_data_VN[:78*2]).reshape(-1, 800)
train_data_V_W = np.append(train_data_V[:540*2], train_data_W[:4*2]).reshape(-1, 800)
train_data_S_VW = np.append(train_data_S[:181*2], train_data_V_W[:181*2])
train_data_S_BBB_VW = np.append(train_data_BBB[:181*2], train_data_S_VW)
train_all_data = np.append(train_data_N[:544*2], train_data_S_BBB_VW)
Train_Data_ALL = train_all_data.reshape(-1, 800)
test_data_N_T = np.append(train_data_normal[202*2 : 253*2], train_data_T[35*2: 44*2])
test_data_St_SN = np.append(train_data_St[37*2: 47*2], train_data_SN[191*2: 239*2])
test_N_except_VN = np.append(test_data_N_T, test_data_St_SN)
test_data_N = np.append(test_N_except_VN, train_data_VN[78*2 : 96*2])
test_data_S_BBB = np.append(train_data_S[543*2:679*2], train_data_BBB[543 * 2])
test_data_V_W = train_data_V[540*2:675*2].copy()
test_data_N_VW = np.append(test_data_N, test_data_V_W)
test_data_ALL = np.append(test_data_N_VW, test_data_S_BBB)
Test_Data_All = test_data_ALL.reshape(-1, 800)

label_data_S = np.ones(543*2)
label_data_Not_S = np.zeros(543*2)
Label_data_train = np.append(label_data_S, label_data_Not_S)

enc = OneHotEncoder()
Label_For_Train = Label_data_train.reshape(-1,1)
enc.fit(Label_For_Train)
Label_for_train =enc.transform(Label_For_Train).toarray()

X_train,X_valid,y_train,y_valid = train_test_split(Train_Data_ALL,Label_for_train,test_size=0.25,random_state= 666)
# print(X_train.shape)

checkpoint = keras.callbacks.ModelCheckpoint('DenseNet_Classfication_2_Whether_N_qrs_dimension_of_1_depth_40_Subjected_Oriented.h5', monitor='val_acc', verbose=1, save_best_only=True)
earlystop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=100, verbose=1, mode='auto')
xtrain = X_train.reshape(-1,COLS,CHANNELS)
xvalid = X_valid.reshape(-1,COLS,CHANNELS)
model.fit(xtrain, y_train, batch_size=batch_size, epochs=nb_epoch,
         validation_data=(xvalid,y_valid),
         callbacks = [earlystop,checkpoint])