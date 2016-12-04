import numpy as np
from keras.layers import Input,merge
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten,Convolution2D,MaxPooling2D
import keras.backend.tensorflow_backend
from keras.callbacks import EarlyStopping, ModelCheckpoint
def model(weights,bias):
    #Input
    x = Input(shape=(None,None,1))
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
    return VDSR_net
