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

  - #### 'N' : 0 , 表示正常，Normal

  - #### 'A'  : 1，表示心房颤动，是临床上最常见的心率失常之一，Atrial Fibrillation

  - #### 'O' : 2，其它节律，Other rhythm

  - #### '~' : 3，噪声，Noisy 

####                    

- #### 频率：300HZ



- #### 标签结果示例：[[0.45259112 0.28135175 0.2653885  0.00066858]]


- ####                            依次表示 N(Normal，正常)、A(AF，Atrial Fibrillation心房颤动)、O(Other rhythm、其它节律)、~(Noisy，噪声)

- #### 4个测试文件分别为AF_1、Noisy_3、Normal_0、Other_2，都放在文件中


