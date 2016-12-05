import modify_weights as utils
import numpy as np
from VDSR_model import model
import scipy.io as sio
import keras.backend.tensorflow_backend as back

matpath = "./Matlab_mat/im_y.mat"
input = sio.loadmat(matpath)['im_y'].reshape(1,256,256,1).astype('float64')
print(input.dtype)

mat_path = "./Matlab_mat/VDSR_Official.mat"
weights,bias = utils.get_original_weights(mat_path) #(3,3,1,64) or (3,3,64,64)
# ratio,weights,bias = utils.get_modify_weights(mat_path)
ratio = 0
print('%d %% kernels were eliminated, but %.1f %% weights were eliminated' %(15,ratio*100))
print('No. , weights\'s shape, bias\'s shape')

for i in range(len(weights)):
    # print(weights.shape)
    weights[i] = weights[i].astype('float64')
    # weights[i] = weights[i].transpose(2,3,1,0)
    bias[i] = bias[i].flatten().astype('float64')
    print("layer %d: %r, %r" %(i,weights[i].shape,bias[i].shape))


NET = model(weights,bias)
answer = NET.predict(input).reshape(256,256)
print(answer.dtype)

sio.savemat('pythonanswer.mat',{'im_h_y':answer})




if back._SESSION:
   import tensorflow as tf
   tf.reset_default_graph() 
   back._SESSION.close()
   back._SESSION = None
