import numpy as np
import modify_weights as utils
import ImagePreprocessing as Ip
from keras.layers import Input,merge
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten,Convolution2D,MaxPooling2D
from keras.optimizers import SGD
import keras.backend.tensorflow_backend
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
from keras import backend as K
import keras.backend.tensorflow_backend
import numpy
import math

mat_path = "./Matlab_mat/model_15.mat"
# mat_path = "./Matlab_mat/VDSR_Official.mat"
##### Get modify weight model ########
ratio,weights,bias = utils.get_modify_weights(mat_path)
print('%d %% kernels were eliminated, but %.1f %% weights were eliminated' %(15,ratio*100))
print('No. , weights\'s shape, bias\'s shape')
for i in range(len(weights)):
    weights[i] = weights[i].transpose(2,3,1,0)
    bias[i] = bias[i].flatten()
    print("layer %d: %r, %r" %(i,weights[i].shape,bias[i].shape))
###### Get Matlab_mat processed data #########
T_data,T_label = utils.get_train_data()
V_data,V_label = utils.get_vali_data()
print(T_data.shape,T_label.shape)
print(V_data.shape,V_label.shape)

######Start to build residual network ########
#Input
x = Input(shape=(41,41,1))
# conv path
conv_y = x
for i in range(len(weights)):
    n_filter = weights[i].shape[3]
    row = weights[i].shape[0]
    col = weights[i].shape[1]
    conv_y  = Convolution2D(n_filter,row,col,activation='relu',
                            weights=[weights[i],bias[i]],border_mode='same')(conv_y)
#short cut
shortcut_y = x
y = merge([shortcut_y, conv_y], mode='sum')
VDSR_net = Model(input=x, output=y)

sgd = SGD(lr=0.001,momentum=0.9,decay=0.0001,clipnorm=0.1)
VDSR_net.compile(optimizer = sgd, loss= 'mse')

# answer = VDSR_net.predict(V_data)
# a = psnr(answer[0],V_label[0])
# print(a)

VDSR_net.fit(T_data,T_label,batch_size=64,nb_epoch=10,verbose=1,validation_data=(V_data,V_label))

if keras.backend.tensorflow_backend._SESSION:
   import tensorflow as tf
   tf.reset_default_graph() 
   keras.backend.tensorflow_backend._SESSION.close()
   keras.backend.tensorflow_backend._SESSION = None
