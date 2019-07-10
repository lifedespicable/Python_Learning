import tensorflow as tf
import os
import numpy as np
import _pickle

CIFAR_DIR = "D:\\算法学习指南\\06.深度学习之神经网络（CNN RNN GAN）算法原理+实战\\课程数据\\cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))


def load_data(filename):
    # read data from data file
    with open(filename, 'rb') as f:
        data = _pickle.load(f)
        return data[b'data'], data[b'labels']


x = tf.placeholder(tf.float32, [None, 3072])

# y是[None]
y = tf.placeholder(tf.int64, [None])

# w是一个(3072 * 1)
w = tf.get_variable('w', [x.get_shape()[-1], 1],
                    initializer=tf.random_normal_initializer(0, 1))

# b是(1,)
b = tf.get_variable('b', [1],
                    initializer=tf.constant_initializer(0.0))

# x[None,3072] , w[3072,1] , x * w = [None,1]
y_ = tf.matmul(x, w) + b

# p_y_1 也是[None,1]
p_y_1 = tf.nn.sigmoid(y_)

# 把y变成[None,1]
y_reshaped = tf.reshape(y,(-1,1))
y_reshaped_float = tf.cast(y_reshaped,tf.float32)

loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))

# predict的类型是bool
predict = p_y_1 > 0.5
# correct_prediction [1,0,1,0,1,1,1,0,1,0,1,0]
correct_prediction = tf.equal(tf.cast(predict,tf.int64),y_reshaped)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))

with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
