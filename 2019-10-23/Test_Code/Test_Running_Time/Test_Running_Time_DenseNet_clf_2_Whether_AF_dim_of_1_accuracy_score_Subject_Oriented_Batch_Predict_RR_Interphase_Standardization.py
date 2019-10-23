from keras.models import *
import os
import numpy as np
from sklearn.model_selection import train_test_split
import json
from sklearn.preprocessing import StandardScaler

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
train_data_Su = load_data('../train_data/Su')
train_data_Wr = load_data('../train_data/Wr')
train_data_Sb = load_data('../train_data/Sb')
train_data_Ab = load_data('../train_data/Ab')
train_data_other = load_data('../train_data/other')
train_data_N_AF = np.append(train_data_normal, train_data_AF)
train_data_Arr_Su = np.append(train_data_Arr, train_data_Su)
train_data_Wr_Sb = np.append(train_data_Wr, train_data_Sb)
train_data_Ab_Other = np.append(train_data_Ab, train_data_other)
trian_N_AF_Arr_Su = np.append(train_data_N_AF, train_data_Arr_Su)
train_Wr_Sb_Ab_Other = np.append(train_data_Wr_Sb, train_data_Ab_Other)
train_data_all = np.append(trian_N_AF_Arr_Su, train_Wr_Sb_Ab_Other)
Test_Data_all = train_data_all.reshape(-1, 6384)

# 拟设定 V_W_qrs 的标签为 0，S_qrs 的标签为 1，BBB_qrs的标签为 2，
# N_qrs的标签为 3，Other_qrs的标签为 4
model_AF = load_model('DenseNet_Classfication_2_Whether_AF_dimension_of_1_subject_oriented_depth_31_RR_Interphase_0.98253.h5')

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

# %%time(此行代码只能在 jupyter notebook 中使用)
Predict_Label = Serial_Test_Accuracy_On_Waveform(Test_Data_all, 6384)