close all;
clear all;


addpath('utils')

im_l = imread('./TEST_data/slena.bmp');
im_gt = imread('./TEST_data/mlena.bmp');
up_scale = 2;
shave = 1;


im_gt = modcrop(im_gt,up_scale);
im_gt = double(im_gt);
im_l  = double(im_l)/ 255.0;

[H,W,C] = size(im_l);
if C == 3
    im_l_ycbcr = rgb2ycbcr(im_l);
else
    im_l_ycbcr = zeros(H,W,C);
    im_l_ycbcr(:,:,1) = im_l;
    im_l_ycbcr(:,:,2) = im_l;
    im_l_ycbcr(:,:,3) = im_l;
end


im_l_y = im_l_ycbcr(:,:,1);

load('python_im_h_y.mat');%load im_h_y

im_h_y = im_h_y * 255;
im_h_ycbcr = imresize(im_l_ycbcr,up_scale,'bicubic');
im_b = ycbcr2rgb(im_h_ycbcr) * 255.0;
im_h_ycbcr(:,:,1) = im_h_y / 255.0;
im_h  = ycbcr2rgb(im_h_ycbcr) * 255.0;

figure;imshow(uint8(im_b));title('Bicubic Interpolation');
figure;imshow(uint8(im_h));title('SCN Reconstruction');
figure;imshow(uint8(im_gt));title('Origin');

if shave == 1;
    shave_border = round(up_scale);
else
    shave_border = 0;
end

sr_psnr = compute_psnr(im_h,im_gt,shave_border);
bi_psnr = compute_psnr(im_b,im_gt,shave_border);
fprintf('sr_psnr: %f dB\n',sr_psnr);
fprintf('bi_psnr: %f dB\n',bi_psnr);

