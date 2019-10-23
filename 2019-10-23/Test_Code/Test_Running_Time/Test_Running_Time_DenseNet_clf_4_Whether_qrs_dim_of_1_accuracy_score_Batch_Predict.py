from keras.models import *
import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 使用第二块GPU（从0开始）

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
train_data_P = load_data('../train_qrs_data/QRS_data/P_qrs')
train_data_St = load_data('../train_qrs_data/QRS_data/st_qrs')
train_data_SN = load_data('../train_qrs_data/QRS_data/S_qrs_')
train_data_VN = load_data('../train_qrs_data/QRS_data/V_qrs_')
train_data_WN = load_data('../train_qrs_data/QRS_data/W_qrs_')
train_data_V = load_data('../train_qrs_data/QRS_data/V_qrs')
train_data_W = load_data('../train_qrs_data/QRS_data/W_qrs')
train_data_S = load_data('../train_qrs_data/QRS_data/S_qrs')
train_data_BBB = load_data('../train_qrs_data/QRS_data/BBB_qrs')
train_data_N_T = np.append(train_data_normal, train_data_T)
train_data_SN_St = np.append(train_data_SN, train_data_St)
train_data_St_N_T_SN = np.append(train_data_N_T, train_data_SN_St)
train_N = np.append(train_data_St_N_T_SN, train_data_VN)
train_V_W = np.append(train_data_V,train_data_W)
train_S_BBB = np.append(train_data_S, train_data_BBB)
train_N_VW = np.append(train_N,train_V_W)
train_all_data = np.append(train_N_VW, train_S_BBB)

# 拟设定 V_W_qrs 的标签为 0，S_qrs 的标签为 1，BBB_qrs的标签为 2，
# N_qrs的标签为 3，Other_qrs的标签为 4
model_clf_4 = load_model('DenseNet_Classfication_4_Whether_qrs_dimension_of_1_depth_16_Subject_Oriented_0.84807.h5')

def Parallel_Test_Accuracy_On_Waveform(X_test, cols):
    X = X_test.reshape(-1, cols, 1)
    Result = []
    label_all = model_clf_4.predict(X)
    for i in range(label_all.shape[0]):
        full_label = []
        label_N = label_all[i][0]
        label_VW = label_all[i][1]
        label_S = label_all[i][2]
        label_BBB = label_all[i][3]
        full_label.append(label_VW)
        full_label.append(label_S)
        full_label.append(label_BBB)
        full_label.append(label_N)
        full_type = np.array(full_label)
        if(label_VW == np.max(full_type)):
            Result.append(1)
        elif(label_S == np.max(full_type)):
            Result.append(2)
        elif(label_BBB == np.max(full_type)):
            Result.append(3)
        else:
            Result.append(0)
    result = np.array(Result)
    return result

# %%time(这行代码只能在Jupyter Notebook中使用)
Result_All = Parallel_Test_Accuracy_On_Waveform(train_all_data, 800)