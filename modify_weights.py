import scipy.io as sio
import numpy as np
def get_train_data():
    matpath1 = "./Matlab_mat/data.mat"
    matpath2 = "./Matlab_mat/label.mat"
    data = sio.loadmat(matpath1)['data'].transpose(3,0,1,2)
    label = sio.loadmat(matpath2)['label'].transpose(3,0,1,2)
    return data,label
def get_vali_data():
    matpath1 = "./Matlab_mat/validata.mat"
    matpath2 = "./Matlab_mat/valilabel.mat"
    validata = sio.loadmat(matpath1)['data'].transpose(3,0,1,2)
    valilabel = sio.loadmat(matpath2)['label'].transpose(3,0,1,2)
    return validata,valilabel
def get_original_weights(path):
    mat_path = path #"./VDSR_15.mat"
    model = sio.loadmat(mat_path)['model'][0][0]
    weights = model[0][0]
    bias = model[1][0]
    weights[19] = weights[19].reshape(3,3,64,1)
    return weights,bias

def get_modify_weights(path): 
    mat_path = path #"./VDSR_15.mat"
    model = sio.loadmat(mat_path)['model'][0][0]
    weights = model[0][0]
    bias = model[1][0]
    weights[19] = weights[19].reshape(3,3,64,1)
    all_para=0
    left_para = 0
    ############################################
    for i  in range(len(weights)):  #20-layers 
        # print(weights[i].shape)
        weights[i] = weights[i].transpose(3,2,0,1)
        # weights[i] = weights[i].astype('float64')
        # print(weights[i].shape)
        # exit(-1)
        a,b,c,d = weights[i].shape
        all_para+=( a*b*c*d)
    for i  in range(len(weights)-1):  #20-layers 
        erase_index = []
        for j in range(len(weights[i])): #filters
            if not weights[i][j].any():
                erase_index.append(j)
        new_this_layer = np.delete(weights[i],erase_index,axis=0)
        new_this_bias  = np.delete(bias[i],erase_index,axis=0) 
        new_next_layer = []
        for k in range(len(weights[i+1])):
            new_kernal = np.delete(weights[i+1][k],erase_index,axis=0)
            new_next_layer.append(new_kernal)
        new_next_layer = np.array(new_next_layer)
        weights[i] = new_this_layer
        bias[i] = new_this_bias
        weights[i+1] = new_next_layer
    for i  in range(len(weights)):  #20-layers 
        a,b,c,d = weights[i].shape
        left_para += (a*b*c*d)

    return (1-left_para/all_para),weights,bias

