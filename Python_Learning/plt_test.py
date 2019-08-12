import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

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
# plt.hist(data) # 只要传入数据，直方图就会统计数据出现的次数，hist是表示的是直方图
# plt.show()

# x = np.arange(1, 10)
# y = x
# fig = plt.figure()
# plt.scatter(x, y, c= 'r', marker= 'o') # c='r'表示散点的颜色为红色，marker表示指定散点的形状为圆形
# plt.show()

# 使用Pandas与Matplotlib进行结合
iris = pd.read_csv('iris_training.csv')
print(iris.head())
print(iris.shape)
print(type(iris))
print(iris.dtypes)

# 绘制散点图
# iris.plot(kind = 'scatter', x = '-0.07415843', y = '-0.115833877')

# 没啥用，只是让Pandas的plot()方法在Pycharm上显示
# plt.show()

# 设置样式
sns.set(style= 'white', color_codes= True)
# 设置绘制格式为散点图
sns.jointplot(x = '120', y = '4', data= iris, height = 5)
# distplot绘制曲线
sns.distplot(iris['120'])
# 没啥用，只是让Pandas的plot()方法在Pycharm上显示
plt.show()

# FacetGrid 一般绘图函数
# hue 彩色显示分类 0/1/2
# plt.scatter 绘制散点图
# add_legend() 显示分类的描述信息
# sns.FacetGrid(iris, hue= 'virginica', height= 5).map(plt.scatter, '-0.07415843', '-0.115833877').add_legend()
# sns.FacetGrid(iris, hue= 'virginica', height= 5).map(plt.scatter, '0.005250311', '-0.062491073').add_legend()
# 没啥用，只是让Pandas的plot()方法在Pycharm上显示
# plt.show()
