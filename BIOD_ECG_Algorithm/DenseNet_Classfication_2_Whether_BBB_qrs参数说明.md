#define DenseNet parms
ROWS = 1
COLS = 800
CHANNELS = 1
nb_classes = 2
batch_size = 20
nb_epoch = 10000
img_dim = (ROWS,COLS,CHANNELS)
densenet_depth = 40
densenet_growth_rate = 10

### 此时验证集准确率为0.83992

需要重点注意的事项

1、测试和训练数据中不能存在同一个人的数据（通过文件名去识别）
2、片段分类拟定先跑出 AF 和非AF 的模型

