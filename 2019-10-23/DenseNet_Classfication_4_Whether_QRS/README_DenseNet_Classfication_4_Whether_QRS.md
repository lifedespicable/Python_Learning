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

model = load_model('DenseNet_4.h5')
pred = model.predict(data)

加载模型，输出预测值

data需要与原始训练格式相同
可逐条预测，也可批量预测



- #### 输入数据格式：1行*9000列


- #### 标签信息：

  - #### 'N' : 0 , 表示窦性

  - #### 'VW'  : 1，表示室性
  
  - #### 'S' : 2，表示房性

  - #### 'BBB' : 3，表示束支阻滞 

####                    

- #### 频率：200HZ



- #### 标签结果示例：[[0.45259112 0.28135175 0.2653885  0.00066858]]


- #### 依次表示 N(Normal，正常)、VW(室性)、S(房性)、BBB(束支阻滞)


