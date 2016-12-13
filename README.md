# VDSR_fine-tune
This is an implementation of "Deep Convolutional Neural Network on iOS Mobile Devices",Chun-Fu (Richard) Chen
,on the topic of VDSR(Very Deep Super Resolution).

The original model and part of my code is copied from ["caffe-vdsr"](https://github.com/huangzehao/caffe-vdsr) , which is an implementation of ["Accurate Image Super-Resolution Using Very Deep Convolutional Networks"](http://cv.snu.ac.kr/research/VDSR/) (CVPR 2016 Oral Paper) 

##Instruction
In order to implement a convolutional computation on mobile device, we need to prune the weights of the model to make the inference process faster,and at the same time,not losing the performance.

This method is trying to eliminate some kernels which are not important, and then fine-tune the model.

## Dependency 
### Train(fine-tune)
- [keras](https://github.com/fchollet/keras)
- [scipy.io](https://www.scipy.org/)
- [Tensorflow](https://www.tensorflow.org/)

### Test
- [MatConvNet](http://www.vlfeat.org/matconvnet/)
- [Matlab](https://www.mathworks.com/products/matlab.html)

## Usgae
###Data
The data I used was preprocessd by Matlab,because I found that the "rgb2ycbcr" and "imresize" functions cannot be reproduced by any Python packages.

So you just need to place these data under the "Matlab_mat" directory.

1. training data & aug training data : [here](https://drive.google.com/file/d/0Bw_IymwywdSnWUo0ZTlkdmtKcVk/view?usp=sharing) and [here](https://drive.google.com/open?id=0Bw_IymwywdSndVFBdDVJWm1kRHc)

2. training labels & aug training labels : [here](https://drive.google.com/open?id=0Bw_IymwywdSnT21QdEZGZGpwMlE) and [here](https://drive.google.com/open?id=0Bw_IymwywdSnZ2xTaXowbHJULXc)

3. validation data & aug validationdata : [here](https://drive.google.com/open?id=0Bw_IymwywdSnbmdvbmtGU0ZBejQ) and [here](https://drive.google.com/open?id=0Bw_IymwywdSnQU9EZEw0LWdtQTA)

4. validation label & aug validation labels : [here](https://drive.google.com/open?id=0Bw_IymwywdSnOFU0VjBuMmlhQlE) and [here](https://drive.google.com/open?id=0Bw_IymwywdSnUDhGNEFWNzB6bDA)

### Train(fine-tune)
1. ˋ$ python3 VDSR_fine_tune.py ˋ

### TEST
Have preprocessed low-resolution lena.png to im_y.mat (in kill_kernel directory)

1. ˋ$ python3 TEST.pyˋ

2. There will be an output "python_im_h_y.mat" , put it in your matlab directory

3. open Matlab 

4. run Demo_PSNR_finetune.m you will see the PSNR

5. You can run Demo_SR_Conv.m from the author's github to see the PSNR of original VDSR version. 
  I list the answer in my PSNR_ans.txt

## Result
###Eliminate 15% kenels (actually eliminate 29.8% parameters)
- The Official PSNR: 36.635908 dB

- My fine-tune model PSNR : **36.207958** dB

- **No** fine-tune PSNR: **32.353143** dB

- Just bicubic :32.723209 dB

###Eliminate 18% kenels (actually eliminate 34.9% parameters)
- The Official PSNR: 36.635908 dB

- My fine-tune model PSNR :**36.093349** dB

- **No** fine-tune PSNR: **32.345002** dB

- Just bicubic :32.723209 dB

###Eliminate 20% kenels (actually eliminate 38.1% parameters)
- The Official PSNR: 36.635908 dB

- My fine-tune model PSNR :36.065349 dB

- No fine-tune PSNR: 32.317988 dB

- Just bicubic :32.723209 dB

## Discover
You can see the matlab_vs_python to see the comparison of lots of functions


