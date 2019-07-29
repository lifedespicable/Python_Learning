#-*- coding:utf-8
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
ROWS = 12
COLS = 3640
CHANNELS = 1
nb_classes = 2
batch_size = 6
nb_epoch = 40
img_dim = (ROWS,COLS,CHANNELS)
densenet_depth = 40
densenet_growth_rate = 10

model = createDenseNet(nb_classes=nb_classes,img_dim=img_dim,depth=densenet_depth,
                  growth_rate = densenet_growth_rate)

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# checkpoint = keras.callbacks.ModelCheckpoint('DenseNet.h5', monitor='val_acc', verbose=1, save_best_only=True)
# earlystop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')
# xtrain = X_train.reshape(-1,ROWS,COLS,CHANNELS)
# xvalid = X_valid.reshape(-1,ROWS,COLS,CHANNELS)
# model.fit(xtrain, y_train, batch_size=batch_size, epochs=nb_epoch,
#          validation_data=(xvalid,y_valid),
#          callbacks = [earlystop,checkpoint])
# #model.summary()
# model = load_model('DenseNet.h5')
# pred = model.predict(xvalid)

model = load_model('DenseNet.h5')
test = np.load(".\\train_1.npy")
xvalid = test.reshape(-1,ROWS,COLS,CHANNELS)
print(model.predict(xvalid))