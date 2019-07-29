代码解释：
ROWS #行数
COLS #列数
CHANNELS #通道数
nb_classes #分类数
batch_size
nb_epoch #迭代轮数
img_dim = (ROWS,COLS,CHANNELS)
densenet_depth #深度
densenet_growth_rate #block增长率

输入：
输入数据的行列数和通道数自由截取，将相应的参数设置相符

输出：
输出分类数自由设定，与标签种类相同

格式：
支持python能读取的文件类型

模型解释：

预测函数在代码结束处

model = load_model('DenseNet.h5')
pred = model.predict(data)

加载模型，输出预测值

data需要与原始训练格式相同
可逐条预测，也可批量预测