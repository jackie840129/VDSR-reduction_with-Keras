import numpy as np
import modify_weights as utils
from keras.layers import Input,merge
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Flatten,Convolution2D,MaxPooling2D
from keras.optimizers import SGD,Adam
import keras.backend.tensorflow_backend
from keras.models import load_model
import sys
from keras import backend as K
import scipy.io as sio
from VDSR_model import model

mat_path = "./Matlab_mat/model_15.mat"
##### Get modify weight model ########
ratio,weights,bias = utils.get_modify_weights(mat_path)
print('%d %% kernels were eliminated, but %.1f %% weights were eliminated' %(15,ratio*100))
for i in range(len(weights)):
    weights[i] = weights[i].transpose(2,3,1,0)
    bias[i] = bias[i].flatten()
###### Get Matlab_mat processed data #########
T_data,T_label = utils.get_train_data()
V_data,V_label = utils.get_vali_data()
print(T_data.shape,T_label.shape)
print(V_data.shape,V_label.shape)

######Start to build residual network ########
VDSR_net = model(weights,bias)
adam = Adam(lr=0.001,clipnorm=0.1)

VDSR_net.compile(optimizer = adam, loss= 'mse')

model_path = "./checkpoint/model.h5" 
mc = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

VDSR_net.fit(T_data,T_label,batch_size=128,nb_epoch=200,verbose=1,validation_data=(V_data,V_label)
             ,callbacks=[mc,es]) 

if keras.backend.tensorflow_backend._SESSION:
   import tensorflow as tf
   tf.reset_default_graph() 
   keras.backend.tensorflow_backend._SESSION.close()
   keras.backend.tensorflow_backend._SESSION = None
