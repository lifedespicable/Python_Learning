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

train_data_normal = load_data('../train_qrs_data/QRS_data/N_qrs')
train_data_T = load_data('../train_qrs_data/QRS_data/T_qrs')
train_data_St = load_data('../train_qrs_data/QRS_data/st_qrs')
train_data_SN = load_data('../train_qrs_data/QRS_data/S_qrs_')
train_data_VN = load_data('../train_qrs_data/QRS_data/V_qrs_')
train_data_WN = load_data('../train_qrs_data/QRS_data/W_qrs_')
train_data_V = load_data('../train_qrs_data/QRS_data/V_qrs')
train_data_W = load_data('../train_qrs_data/QRS_data/W_qrs')
train_data_S = load_data('../train_qrs_data/QRS_data/S_qrs')
train_data_BBB = load_data('../train_qrs_data/QRS_data/BBB_qrs')
train_N_T = np.append(train_data_normal[:202*2], train_data_T[:35*2])
train_St_SN = np.append(train_data_St[:37*2],train_data_SN[:191*2])
train_N_except_VN =  np.append(train_N_T, train_St_SN)
train_data_N = np.append(train_N_except_VN, train_data_VN[:77*2]).reshape(-1, 800)
train_data_V_W = np.append(train_data_V[:540*2], train_data_W[:4*2]).reshape(-1, 800)
train_data_N_VW = np.append(train_data_N[:181*2], train_data_V_W[:181*2])
train_data_S_N_VW = np.append(train_data_S[:181*2], train_data_N_VW)
train_all_data = np.append(train_data_BBB[:543*2], train_data_S_N_VW)
Train_Data_ALL = train_all_data.reshape(-1, 800)
test_data_N_T = np.append(train_data_normal[202*2 : 253*2], train_data_T[35*2: 44*2])
test_data_St_SN = np.append(train_data_St[37*2: 47*2], train_data_SN[191*2: 239*2])
test_N_except_VN = np.append(test_data_N_T, test_data_St_SN)
test_data_N = np.append(test_N_except_VN, train_data_VN[78*2 : 96*2]).reshape(-1, 800)[:45*2]
test_data_S = train_data_S[543*2:679*2].reshape(-1, 800)[:136*2]
test_data_BBB = train_data_BBB[543 * 2:].reshape(-1, 800)[:45*2]
test_data_V_W = train_data_V[540*2:675*2].copy().reshape(-1, 800)
test_data_N_VW = np.append(test_data_N, test_data_V_W[:46*2])
test_data_except_S = np.append(test_data_N_VW, test_data_BBB)
test_data_ALL = np.append(test_data_S, test_data_except_S)
Test_Data_All = test_data_ALL.reshape(-1, 800)

label_data_S = np.ones(136*2)
label_data_Not_S = np.zeros(136*2)
Label_data_train = np.append(label_data_S, label_data_Not_S)

# 拟设定 V_W_qrs 的标签为 0，S_qrs 的标签为 1，BBB_qrs的标签为 2，
# N_qrs的标签为 3
model_V_W = load_model('DenseNet_Classfication_2_Whether_V_W_qrs_dimesion_of_1_0.96985.h5')
model_S = load_model('DenseNet_Classfication_2_Whether_S_qrs_dimension_of_1_depth_40_Subjected_Oriented_0.93186.h5')
model_BBB = load_model('DenseNet_Classfication_2_Whether_BBB_qrs_dimension_1_0.85780.h5')
model_N = load_model('DenseNet_Classfication_2_Whether_N_qrs_dimension_of_1_0.93270.h5')

def Serial_Test_Accuracy_On_Waveform(X_test, cols):
    X = X_test.reshape(-1, cols, 1)
    Result = []
    label_S = model_S.predict(X)
    for i in range(label_S.shape[0]):
        if(label_S[i][1] > label_S[i][0]):
            Result.append(1)
        else:
            Result.append(0)
    result = np.array(Result)
    return result

Predict_Label = Serial_Test_Accuracy_On_Waveform(Test_Data_All, 800)

def accuracy_score(True_Label, Predict_Label):
    return sum(True_Label == Predict_Label) / len(True_Label)

accuracy_score(Label_data_train, Predict_Label)

