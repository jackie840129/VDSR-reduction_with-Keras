import numpy as np
from keras.layers import Input,merge
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten,Convolution2D,MaxPooling2D
import keras.backend.tensorflow_backend
from keras.callbacks import EarlyStopping, ModelCheckpoint
def model(weights,bias):
    # temp_list=[]
    
    #Input
    x = Input(shape=(None,None,1))
    # conv path
    conv_y = x
    ##first layer
    n_filter = weights[0].shape[3]
    row = weights[0].shape[0]
    col = weights[0].shape[1]
    conv_y  = Convolution2D(n_filter,row,col,
                             weights=[weights[0],bias[0]],border_mode='same')(conv_y)
    for i in range(len(weights)-1):
        n_filter = weights[i+1].shape[3]
        row = weights[i+1].shape[0]
        col = weights[i+1].shape[1]
        conv_y  = Activation('relu')(conv_y)
        conv_y  = Convolution2D(n_filter,row,col,
                                weights=[weights[i+1],bias[i+1]],border_mode='same')(conv_y)
        # temp_list.append(Model(input=x,output=conv_y))
    #short cut
    shortcut_y = x
    y = merge([shortcut_y, conv_y], mode='sum')
    VDSR_net = Model(input=x, output=y)
    return VDSR_net
