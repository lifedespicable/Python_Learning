from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import _pickle
import numpy as np
import os

CIFAR_DIR = "D:\\算法学习指南\\06.深度学习之神经网络（CNN RNN GAN）算法原理+实战\\课程数据\\cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))

with open(os.path.join(CIFAR_DIR, "data_batch_1"), 'rb') as f:
    data = _pickle.load(f, encoding='bytes')
    print(type(data))
    for i in data.keys():
        print(i)
    print(type(data[b'batch_label']))
    print(type(data[b'labels']))
    print(type(data[b'data']))
    print(type(data[b'filenames']))
    print(data[b'data'].shape)
    print(data[b'data'][0:2])
    print(data[b'labels'][0:2])
    print(data[b'batch_label'])
    print(data[b'filenames'][0:2])

# 32 * 32 = 1024 1024 * 3 = 3072
# RR--GG-BB = 3072

image_arr = data[b'data'][100]
image_arr = image_arr.reshape(3, 32, 32)
image_arr = image_arr.transpose(1, 2, 0)

# %matplotlib inline
imshow(image_arr)
plt.show()
