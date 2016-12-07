import modify_weights as utils
import numpy as np
from VDSR_model import model
import scipy.io as sio
import keras.backend.tensorflow_backend as back

matpath = "./Matlab_mat/im_y.mat"
im_y = sio.loadmat(matpath)['im_y'].reshape(1,256,256,1)
print("#######input: im_y's dtype: ",im_y.dtype)
####### ↑ same as matlab ##########

mat_path = "./Matlab_mat/VDSR_Official.mat"
ratio,weights,bias = utils.get_original_weights(mat_path) #(3,3,1,64) or (3,3,64,64)
# ratio,weights,bias = utils.get_modify_weights(mat_path)
print('#######%d %% kernels were eliminated, but %.1f %% weights were eliminated' %(15,ratio*100))
print('#######No. , weights\'s shape, bias\'s shape')
print("#######Model weights 's dtype: ",weights[0].dtype)
print("#######Model[0]'s shape:",weights[0].shape,"row,col,channel,output")
####### ↑ same as matlab ##########
## Python
#[[[[ -4.07601194e-03   7.99347162e-02   1.14039838e-01]
#   [ -1.69451848e-01   2.60733843e-01   9.97218937e-02]
#   [ -8.69311094e-02  -1.40367508e-01  -9.31298658e-02]]]
## Matlab
#           -0.0041    0.0799    0.1140
#           -0.1695    0.2607    0.0997
#           -0.0869   -0.1404   -0.0931


for i in range(len(weights)):
    # print(weights.shape)
    # weights[i] = weights[i].transpose(2,3,1,0)
    bias[i] = bias[i].flatten()
    # print("layer %d: %r, %r" %(i,weights[i].shape,bias[i].shape))


NET,layers = model(weights,bias)
# print(len(layers))
# layer1 = layers[1].predict(im_y)
# print(layer1.reshape(256,256,64).transpose(2,0,1)[0])
#[[ 0.2699566   0.13778641  0.13802331 ...,  0.15392331  0.15386713
#     0.15095545]
#  [ 0.37011145  0.23592564  0.23630566 ...,  0.22676649  0.22686322
#      0.17971549]
#  [ 0.37083319  0.23629616  0.23615094 ...,  0.2296454   0.23164405
#      0.18380305]

answer = NET.predict(im_y).reshape(256,256)
print(answer.dtype)

sio.savemat('pythonanswer.mat',{'im_h_y':answer})




if back._SESSION:
   import tensorflow as tf
   tf.reset_default_graph() 
   back._SESSION.close()
   back._SESSION = None
