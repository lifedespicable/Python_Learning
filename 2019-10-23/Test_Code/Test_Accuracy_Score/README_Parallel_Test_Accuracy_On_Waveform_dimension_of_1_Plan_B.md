### 此为用 4 个二分类来模拟一个四分类的波形判断的测试准确率的代码

### 此种方法是采用的并行测试，将数据放入 4 个模型中进行预测，每一个的预测结果是该输出结果乘以该模型的测试集的准确率，最后确定为是的值最大的那一个模型的结果为输出结果