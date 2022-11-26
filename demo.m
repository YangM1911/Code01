clear;clc;close all;
addpath(genpath('cho_code'));
addpath(genpath('Skeleton1'));
addpath(genpath('pan_code'));
addpath(genpath('Skeleton2'));
addpath(genpath('images'));
addpath(genpath('pyramid'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% set the parameters
fprintf('setting the parameters...\n');
% set the source and result file names and addresses
num = 3;
textImageName = strcat('text',string(num));
textImageAddress = strcat('images/text_images/', textImageName, '.jpg');
KernelName = 'kernel2';
kernelImageAddress = strcat('images/kernel_images/', KernelName, '.png');
latentKernel  =   strcat ('images/results/', textImageName, '_kernel_Multi-scale_fusion.jpg');
latentImage  = strcat ('images/results/', textImageName, '_denoised.jpg');
% iterations and kernel size
opts.prescale = 1; %%downsampling
opts.xk_iter = 5; %% the iterations
opts.k_thresh = 20; 
% denoising parameters
ker_denoised = 1;       % denoising or not
skeleton_method = 1;    % skeleton detection method
threshold = 0.3;        % local threshold  
threshold_all = 0.1;    % global threshold
% model related parameters
gammaL_pixel = 4e-3;
gammaL_grad = 4e-3;
lambda_tv = 0.0022; 
lambda_l0 = 1e-3; 
weight_ring = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if num == 1
    im_blur = imread(textImageAddress);
    im_org = im_blur;
    opts.kernel_size = 53; % kernel size
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    y = im2double(im_blur);
    y_1 = y(:,:,1);
    y_2 = y(:,:,2);
    y_3 = y(:,:,3);
    %% kernel estimation
    [new_ker1,new_ker2,new_ker3] = kernel_estimation_1(y_1,y_2,y_3, ...
    gammaL_pixel, gammaL_grad, opts,threshold,threshold_all,skeleton_method);
    %% Kernel Multi-scale Fusion
    R = multi_scale_fusion(new_ker1,new_ker2,new_ker3);
    %% kernel denoising
    h_R = double(R)/255;  
    kernel = h_R/(sum(sum(h_R))); 
    k = kernel - min(kernel(:));
    k = k./max(k(:));
    denoised_k = kernel_denoised(k,threshold,threshold_all,skeleton_method);
    %% deblurring
    Latent = ringing_artifacts_removal_2(y,denoised_k, lambda_tv, lambda_l0, weight_ring,im_org); 
    for t = 1:3
        q(:,:,t) = guidedfilter(Latent(:,:,t), Latent(:,:,t), 24, 1e-4);
    end
    Latent_image = uint8(255*q);
    imwrite(Latent_image ,latentImage);
    imwrite(mat2gray(denoised_k),latentKernel);

elseif num == 10
    im_blur = imread(textImageAddress);
    im_org = im_blur;
    opts.kernel_size = 75; % kernel size
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    y = im2double(im_blur);
    y_1 = y(:,:,1);
    y_2 = y(:,:,2);
    y_3 = y(:,:,3);
    %% kernel estimation
    [new_ker1,new_ker2,new_ker3] = kernel_estimation_1(y_1,y_2,y_3, ...
    gammaL_pixel, gammaL_grad, opts,threshold,threshold_all,skeleton_method);
    %% Kernel Multi-scale Fusion
    R = multi_scale_fusion(new_ker1,new_ker2,new_ker3);
    %% kernel denoising
    h_R = double(R)/255;  
    kernel = h_R/(sum(sum(h_R))); 
    k = kernel - min(kernel(:));
    k = k./max(k(:));
    denoised_k = kernel_denoised(k,threshold,threshold_all,skeleton_method);
    %% deblurring
    Latent = ringing_artifacts_removal(y,denoised_k, lambda_tv, lambda_l0, weight_ring,im_org); 
    for t = 1:3
        q(:,:,t) = guidedfilter(Latent(:,:,t), Latent(:,:,t), 24, 1e-4);
    end
    Latent_image = uint8(255*q);
    imwrite(Latent_image ,latentImage);
    imwrite(mat2gray(denoised_k),latentKernel);

else
    %% text image
    im_org = imread(textImageAddress);
    %% kernel
    h_org = imread(kernelImageAddress);
    h_org_HD = double((h_org))/255; 
    h_org_HD_1 = double((h_org))/255;
    h = h_org_HD/(sum(sum(h_org_HD))); %Normalization
    kernel_size = size(h,1);
    %% blur text image 
    im_blur = imfilter(im_org, h,'circular');  
    opts.kernel_size = kernel_size; % kernel size
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    y = im2double(im_blur);
    y_1 = y(:,:,1);
    y_2 = y(:,:,2);
    y_3 = y(:,:,3);
    %% kernel estimation
    [new_ker1,new_ker2,new_ker3] = kernel_estimation_2(y_1,y_2,y_3,h_org_HD_1, ...
    gammaL_pixel, gammaL_grad, opts,threshold,threshold_all,skeleton_method);
    %% Kernel Multi-scale Fusion
    R = multi_scale_fusion(new_ker1,new_ker2,new_ker3);
    %% kernel denoising
    h_R = double(R)/255;  
    kernel = h_R/(sum(sum(h_R))); 
    k = kernel - min(kernel(:));
    k = k./max(k(:));
    denoised_k = kernel_denoised(k,threshold,threshold_all,skeleton_method);
    %% deblurring
    Latent = ringing_artifacts_removal(y,denoised_k, lambda_tv, lambda_l0, weight_ring,im_org); 
    for t = 1:3
        q(:,:,t) = guidedfilter(Latent(:,:,t), Latent(:,:,t), 24, 1e-4);
    end
    Latent_image = uint8(255*q);
    imwrite(Latent_image ,latentImage);
    imwrite(mat2gray(denoised_k),latentKernel);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
