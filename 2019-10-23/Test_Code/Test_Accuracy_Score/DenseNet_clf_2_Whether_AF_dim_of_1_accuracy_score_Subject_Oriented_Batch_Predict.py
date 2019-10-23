from keras.models import *
import os
import numpy as np
from sklearn.model_selection import train_test_split

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 使用第二块GPU（从0开始）

def load_data(path):
    # read data from data file
    temp_data = []
    namelist = [x for x in os.listdir(path)]
    for i in range( len(namelist) ):
        if namelist[i].endswith('npy'):
            temp_data_one = np.load(path + '/' + namelist[i])
            temp_data.append(temp_data_one[0])
            temp_data.append(temp_data_one[1])
    train_data = np.array(temp_data)
    return train_data

train_data_normal = load_data('../train_data/N')
train_data_AF = load_data('../train_data/AF')
train_data_Arr = load_data('../train_data/Arr')
train_data_Ab = load_data('../train_data/Ab')
train_data_other = load_data('../train_data/other')
true_data_Ab = train_data_Ab[:3*2]
test_data_Ab = train_data_Ab[3*2:4*2]
true_data_AF = train_data_AF[:229*2]
test_data_AF = train_data_AF[229*2:]
true_data_Arr = train_data_Arr[:25*2]
test_data_Arr = train_data_Arr[25*2 : 31*2]
true_data_N  = train_data_normal[:60*2]
test_data_N  = train_data_normal[60*2 : 75*2]
true_data_other = train_data_other[:141*2]
test_data_other = train_data_other[141*2: 176*2]
true_data_Ab_Arr = np.append(true_data_Ab, true_data_Arr)
true_data_N_Other = np.append(true_data_N, true_data_other)
true_data_Not_AF = np.append(true_data_Ab_Arr, true_data_N_Other)
true_data = np.append(true_data_AF, true_data_Not_AF)
test_data_Ab_Arr = np.append(test_data_Ab, test_data_Arr)
test_data_N_Other = np.append(test_data_N, test_data_other)
test_data_Not_AF = np.append(test_data_Ab_Arr, test_data_N_Other)
test_data_all = np.append(test_data_AF, test_data_Not_AF)
Train_Data_all = true_data.reshape(-1, 6000)
Test_Data_all = test_data_all.reshape(-1, 6000)

Label_test_AF = np.ones(114)
Label_test_Not_AF = np.zeros(114)
Label_data_test = np.append(Label_test_AF, Label_test_Not_AF)

# 拟设定 V_W_qrs 的标签为 0，S_qrs 的标签为 1，BBB_qrs的标签为 2，
# N_qrs的标签为 3，Other_qrs的标签为 4
model_AF = load_model('DenseNet_Classfication_2_Whether_AF_dimension_of_1_subject_oriented_0.72926_depth_4.h5')

def Serial_Test_Accuracy_On_Waveform(X_test, cols):
    X = X_test.reshape(-1, cols, 1)
    Result = []
    label_AF = model_AF.predict(X)
    for i in range(label_AF.shape[0]):
        if(label_AF[i][1] > label_AF[i][0]):
            Result.append(1)
        else:
            Result.append(0)
    result = np.array(Result)
    return result

Predict_Label = Serial_Test_Accuracy_On_Waveform(Test_Data_all, 6000)

def accuracy_score(True_Label, Predict_Label):
    return sum(True_Label == Predict_Label) / len(True_Label)

accuracy_score(Label_data_test, Predict_Label)