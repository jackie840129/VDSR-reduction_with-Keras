import numpy as np
import keras.backend.tensorflow_backend
from keras.models import load_model
import scipy.io as sio


model_path = "./checkpoint/model.h5"
VDSR_net = load_model(model_path)

matpath = "./Matlab_mat/im_y.mat"
im_y = sio.loadmat(matpath)['im_y'].reshape(1,256,256,1)
print("#######input: im_y's dtype: ",im_y.dtype)
answer = VDSR_net.predict(im_y).reshape(256,256)

answer_path = "python_im_h_y.mat"
sio.savemat(answer_path,{'im_h_y':answer})



if keras.backend.tensorflow_backend._SESSION:
   import tensorflow as tf
   tf.reset_default_graph() 
   keras.backend.tensorflow_backend._SESSION.close()
   keras.backend.tensorflow_backend._SESSION = None
