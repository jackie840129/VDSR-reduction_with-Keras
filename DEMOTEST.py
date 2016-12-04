import PIL.Image as im
import numpy as np
import os
import scipy.misc as misc
import skimage.io as sio
from VDSR_model import model
import modify_weights as utils
from modcrop import modcrop
import skimage.transform as sf
# mat_path = "./model_15.mat"
mat_path = "./Matlab_mat/VDSR_Official.mat"
ratio,weights,bias = utils.get_modify_weights(mat_path)
print('%d %% kernels were eliminated, but %.1f %% weights were eliminated' %(15,ratio*100))
print('No. , weights\'s shape, bias\'s shape')
for i in range(len(weights)):
    weights[i] = weights[i].transpose(2,3,1,0)
    bias[i] = bias[i].flatten()
    print("layer %d: %r, %r" %(i,weights[i].shape,bias[i].shape))

im_l = sio.imread('TEST_data/slena.bmp')
im_gt = sio.imread('TEST_data/mlena.bmp')  #high resolution
im_gt = modcrop(im_gt,2)

## conver to PIL image and transfer to YCBCR
## then transform to numpy array
image = im.fromarray(im_l)
im_l_ycbcr = image.convert('YCbCr')
im_l_ycbcr = np.ndarray((image.size[1], image.size[0], 3), 'u1', im_l_ycbcr.tobytes())
print(im_l_ycbcr[0][0])

##output: [134 95 172]
## Matlab output: [132 99 167]

## TEST imresize for type: uint8 ##
im_lx2 = misc.imresize(im_l,200,'bicubic')
print(im_lx2.shape)
print(im_lx2[0][0])

##output [197 114 77]
##Matlab output [197 114 77]

## TEST float####
im_l_f = im_l.astype('float32')
im_lx2 = misc.imresize(im_l_f,200,'bicubic')
print(im_lx2[0][0])
## output [193 100 59]
## Matlab output [196.9852 114.0654 76.8445]


##### TEST skimage.transform.resize ####
im_lx2 = sf.resize(im_l,(256,256))
print(im_lx2.shape,im_lx2[0][0])
##output [0.434 0.2514 0.1698]
## Matlab output [0.7725 0.4473 0.3014]
