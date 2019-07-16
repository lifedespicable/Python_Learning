import tensorflow as tf
import os
import numpy as np
import _pickle

CIFAR_DIR = "D:\\算法学习指南\\06.深度学习之神经网络（CNN RNN GAN）算法原理+实战\\课程数据\\cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))


def load_data(filename):
    # read data from data file
    with open(filename, 'rb') as f:
        data = _pickle.load(f,encoding='bytes')
        return data[b'data'], data[b'labels']

class CifarData:
    def __init__(self,filenames,need_shuffle):
        all_data = []
        all_labels = []
        for filename in filenames:
            data,labels = load_data(filename)
            for item,label in zip(data,labels):
                if label in [0,1]:
                    all_data.append(item)
                    all_labels.append(label)
        self._data = np.vstack(all_data)
        self._labels = np.hstack(all_labels)
        print(self._data.shape)
        print(self._labels.shape)
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        # [0,1,2,3,4,5] -> [5,3,4,2,0,1]
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self,batch_size):
        """return batch_size examples as a batch."""
        end_indicator =self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")
        batch_data = self._data[self._indicator: end_indicator]
        batch_labels =self._labels[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels

train_filenames = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1,6)]
test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]

train_data = CifarData(train_filenames, True)


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

init = tf.global_variables_initializer()
# with tf.Session as sess:
    # sess.run([loss,accuracy,train_op],feed_dict={x:,y:})
