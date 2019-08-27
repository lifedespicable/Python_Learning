from keras.models import *
import os
import numpy as np

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

def Serial_Test_Accuracy_On_Waveform(xvalid):
    label_V_W = model_V_W.predict(xvalid)
    label_S = model_S.predict(xvalid)
    label_BBB = model_BBB.predict(xvalid)
    label_N = model_N.predict(xvalid)
    label_Other = model_Other.predict(xvalid)
    if(label_V_W[0, 1] > label_V_W[0, 0]):
        return 0
    elif(label_S[0, 1] > label_S[0, 0]):
        return 1
    elif(label_BBB[0, 1] > label_BBB [0, 0]):
        return 2
    elif(label_N[0, 1] > label_N[0, 0]):
        return 3
    else:
        return 4

def Reshape_To_Train_Data(nparray, rows, cols):
    return nparray.reshape(-1, rows, cols, 1)

def Result(train_data, rows, cols):
    Result = []
    for i in range(train_data.shape[0]):
        xvalid = train_data[i].reshape(-1, rows, cols, 1)
        Result.append(Serial_Test_Accuracy_On_Waveform(xvalid))
    result = np.array(Result)
    return result

def Accuracy(True_Label, Predict_Label):
    j = 0.0
    sum = True_Label.size
    for i in range(True_Label.size):
        if (True_Label[i] == Predict_Label[i]):
            j += 1
    return j/float(sum)

# 拟设定 V_W_qrs 的标签为 0，S_qrs 的标签为 1，BBB_qrs的标签为 2，
# N_qrs的标签为 3，Other_qrs的标签为 4
model_V_W = load_model('DenseNet_Classfication_2_Whether_V_W_qrs_0.92457.h5')
model_S = load_model('DenseNet_Classfication_2_Whether_S_qrs_0.91514.h5')
model_BBB = load_model('DenseNet_Classfication_2_Whether_BBB_qrs_0.83992.h5')
model_N = load_model('DenseNet_Classfication_2_Whether_N_qrs_0.75319.h5')
model_Other = load_model('DenseNet_Classfication_2_Whether_Other_qrs_0.71107.h5')

train_data_normal = load_data('../train_qrs_data/selectedata4s_qrs/N_qrs')
train_data_BBB = load_data('../train_qrs_data/selectedata4s_qrs/BBB_qrs')
train_data_P = load_data('../train_qrs_data/selectedata4s_qrs/P_qrs')
train_data_St = load_data('../train_qrs_data/selectedata4s_qrs/st_qrs')
train_data_Stb = load_data('../train_qrs_data/selectedata4s_qrs/Stb_qrs')
train_data_T = load_data('../train_qrs_data/selectedata4s_qrs/T_qrs')
train_data_V = load_data('../train_qrs_data/selectedata4s_qrs/V_qrs')
train_data_W = load_data('../train_qrs_data/selectedata4s_qrs/W_qrs')
train_data_Other = load_data('../train_qrs_data/selectedata4s_qrs/other_qrs')
train_data_S = load_data('../train_qrs_data/selectedata4s_qrs/S_qrs')

Train_data_normal = Reshape_To_Train_Data(train_data_normal, 1, 800)
Train_data_BBB = Reshape_To_Train_Data(train_data_BBB, 1, 800)
Train_data_P = Reshape_To_Train_Data(train_data_P, 1, 800)
Train_data_St = Reshape_To_Train_Data(train_data_St, 1, 800)
Train_data_Stb = Reshape_To_Train_Data(train_data_Stb, 1, 800)
Train_data_T = Reshape_To_Train_Data(train_data_T, 1, 800)
Train_data_V = Reshape_To_Train_Data(train_data_V, 1, 800)
Train_data_W = Reshape_To_Train_Data(train_data_W, 1, 800)
Train_data_Other = Reshape_To_Train_Data(train_data_Other, 1, 800)
Train_data_S = Reshape_To_Train_Data(train_data_S, 1, 800)

result_N = Result(Train_data_normal, 1, 800)
result_BBB = Result(Train_data_BBB, 1, 800)
result_P = Result(Train_data_P, 1, 800)
result_St = Result(Train_data_St, 1, 800)
result_Stb = Result(Train_data_Stb, 1, 800)
result_T = Result(Train_data_T, 1, 800)
result_V = Result(Train_data_V, 1, 800)
result_W = Result(Train_data_W, 1, 800)
result_Other = Result(Train_data_Other, 1, 800)
result_S = Result(Train_data_S, 1, 800)

V_W_array = np.zeros(2828, dtype = int)
S_array = np.ones(7110 ,dtype = int)
BBB_array = np.full(shape = 884, fill_value = 2, dtype = int)
N_array = np.full(shape = 9392, fill_value = 3, dtype = int)
Other_array = np.full(shape = 8132, fill_value = 4, dtype = int)

True_Label = np.concatenate([V_W_array, S_array, BBB_array, N_array, Other_array])

Predict_Label = np.concatenate([result_V, result_W, result_S, result_BBB,
                                result_N, result_P, result_St, result_Stb,
                                result_T, result_Other])

Accuracy(True_Label, Predict_Label)