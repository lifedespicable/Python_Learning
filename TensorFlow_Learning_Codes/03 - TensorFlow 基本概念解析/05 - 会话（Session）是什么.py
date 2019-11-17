# import tensorflow as tf
# # 创建数据流图：z = x * y
# x = tf.placeholder(tf.float32, name = 'x')
# y = tf.placeholder(tf.float32, name = 'y')
# z = tf.multiply(x, y, name = 'z')
# # 创建会话
# sess = tf.Session()
# # 向数据节点 x 和 y 分别填充浮点数 3.0 和 2.0，并输出结果
# print(sess.run(z, feed_dict= {x: 3.0, y: 2.0}))

# import tensorflow as tf
# # 创建数据流图：c = a + b
# a = tf.constant(1.0, name= 'a')
# b = tf.constant(2.0, name= 'b')
# c = tf.add(a, b, name= 'c')
# # 创建会话
# sess = tf.Session()
# # 估算张量 c 的值
# print(sess.run(c))

import tensorflow as tf
# 创建数据流图： y = W * x + b,其中 W 和 b 为存储节点， x 为数据节点
x = tf.placeholder(tf.float32)
W = tf.Variable(1.0)
b = tf.Variable(1.0)
y = W * x + b
with tf.Session() as sess:
    tf.global_variables_initializer().run()   # Operation.run
    fetch = y.eval(feed_dict = {x: 3.0})      # Tensor.eval
    print(fetch)                              # fetch = 1.0 * 3.0 + 1.0
