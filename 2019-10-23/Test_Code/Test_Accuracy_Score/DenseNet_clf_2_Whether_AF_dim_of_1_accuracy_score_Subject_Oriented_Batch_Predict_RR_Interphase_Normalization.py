from keras.models import *
import os
import numpy as np
from sklearn.model_selection import train_test_split
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 使用第二块GPU（从0开始）

def load_data(path):
    # read data from data file
    temp_data = []
    namelist = [x for x in os.listdir(path)]
    for i in range( len(namelist) ):
        if namelist[i].endswith('npy'):
            temp_data_one = np.load(path + '/' + namelist[i])
            labellist_dir = [x for x in os.listdir('../label_data/')]
            can_break = False
            for j in range( len(labellist_dir) ):
                if can_break:
                    break
                labellist = [x for x in os.listdir('../label_data/' + labellist_dir[j] + '/' +'label_data/'+ 'ecg_data' + '/')]
                for k in range( len(labellist) ):
                    if can_break:
                        break
                    if labellist[k][0:-4] in namelist[i]:
                        with open('../label_data/' + labellist_dir[j] + '/' +'label_data/'+ 'ecg_data' + '/' + labellist[k], 'r') as file:
                            data_json = json.load(file)
                        R_index = []
                        rr = [float(i) for i in data_json['RR']]  # 字符串转数字
                        diff = list(map(lambda x: x[0]-x[1], zip(rr[1:], rr[0:-1])))
                        diff2 = list(map(lambda x: x[0]-x[1], zip(diff[1:], diff[0:-1])))
                        R_index.extend(rr)
                        R_index.extend([0]*(128 - len(rr)))
                        R_index.extend(diff)
                        R_index.extend([0]*(128 - len(diff)))
                        R_index.extend(diff2)
                        R_index.extend([0]*(128 - len(diff2)))
                        can_break = True
            temp_data.append(np.concatenate([temp_data_one[0], np.array(R_index)], axis=0))
            temp_data.append(np.concatenate([temp_data_one[1], np.array(R_index)], axis=0))
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
test_data_AF = train_data_AF[229*2:286*2]
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
Train_Data_all = true_data.reshape(-1, 6384)
Test_Data_all = test_data_all.reshape(-1, 6384)

X1, X2, X3, X4 = np.split(Train_Data_all, [6000, 6128, 6256], axis = 1)
X5, X6, X7, X8 = np.split(Test_Data_all, [6000, 6128, 6256], axis = 1)

def normalization(X_train):
    for i in range(X_train.shape[0]):
        X_train[i] = (X_train[i] - np.min(X_train[i])) / (np.max(X_train[i]) - np.min(X_train[i]))
    return X_train

X1_normalization = normalization(X1)
X2_normalization = normalization(X2)
X3_normalization = normalization(X3)
X4_normalization = normalization(X4)
X5_normalization = normalization(X5)
X6_normalization = normalization(X6)
X7_normalization = normalization(X7)
X8_normalization = normalization(X8)

Train_Data_all = np.hstack([X1, X2, X3, X4])
Test_Data_all = np.hstack([X5, X6, X7, X8])

Label_test_AF = np.ones(114)
Label_test_Not_AF = np.zeros(114)
Label_data_test = np.append(Label_test_AF, Label_test_Not_AF)

# 拟设定 V_W_qrs 的标签为 0，S_qrs 的标签为 1，BBB_qrs的标签为 2，
# N_qrs的标签为 3，Other_qrs的标签为 4
model_AF = load_model('DenseNet_Classfication_2_Whether_AF_dimension_of_1_subject_oriented_depth_10_RR_Interphase_Normalization_0.98253.h5')

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

Predict_Label = Serial_Test_Accuracy_On_Waveform(Test_Data_all, 6384)

def accuracy_score(True_Label, Predict_Label):
    return sum(True_Label == Predict_Label) / len(True_Label)

accuracy_score(Label_data_test, Predict_Label)

confusion_matrix(Label_data_test, Predict_Label)

precision_score(Label_data_test, Predict_Label)

recall_score(Label_data_test, Predict_Label)

f1_score(Label_data_test, Predict_Label)

