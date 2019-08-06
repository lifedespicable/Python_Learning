import matplotlib.pyplot as plt
import numpy as np

# 绘制简单的曲线
# plt.plot([1, 3, 5], [4, 8, 10])
# plt.show()   # 这条命令没有作用，只是单纯让曲线在Pycharm当中显示
#
# x = np.linspace(-np.pi, np.pi, 100)  # x的定义域为-3.14 ~ 3.14，中间间隔100个元素
# plt.plot(x, np.sin(x))
# # 显示所画的图
# plt.show()

# x = np.linspace(-np.pi * 2, np.pi * 2, 100)  # x的定义域为-2pi ~ 2pi，中间间隔100个元素
# plt.figure(1, dpi= 50)  # 创建图表1
# for i in range(1, 5):   # 画四条线
#     plt.plot(x, np.sin(x / i))
# plt.show()

# plt.figure(1, dpi= 50) # 创建图表1，dpi代表图片精细度，dpi越大图片文件越大，杂志要300以上
# data = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 6, 4]
# plt.hist(data) # 只要传入数据，直方图就会统计数据出现的次数
# plt.show()

# x = np.arange(1, 10)
# y = x
# fig = plt.figure()
# plt.scatter(x, y, c= 'r', marker= 'o') # c='r'表示散点的颜色为红色，marker表示指定散点的形状为圆形
# plt.show()


