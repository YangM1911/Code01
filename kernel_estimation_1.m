function [new_ker1,new_ker2,new_ker3] = kernel_estimation_1(y_1,y_2,y_3, gammaL_pixel, gammaL_grad, opts,threshold,threshold_all,skeleton_method);


[kernel1, interim_latent1] = blind_deconv(y_1, gammaL_pixel, gammaL_grad, opts);
k1 = kernel1 - min(kernel1(:));
k1 = k1./max(k1(:));
denoised_k1 = kernel_denoised(k1,threshold,threshold_all,skeleton_method);
new_ker1 = mat2gray(denoised_k1);
new_ker1 = uint8(255*new_ker1);

[kernel2, interim_latent2] = blind_deconv(y_2, gammaL_pixel, gammaL_grad, opts);
k2 = kernel2 - min(kernel2(:));
k2 = k2./max(k2(:));
denoised_k2 = kernel_denoised(k2,threshold,threshold_all,skeleton_method);
new_ker2 = mat2gray(denoised_k2);
new_ker2 = uint8(255*new_ker2);

[kernel3, interim_latent3] = blind_deconv(y_3, gammaL_pixel, gammaL_grad, opts);
k3 = kernel3 - min(kernel3(:));
k3 = k3./max(k3(:));
denoised_k3 = kernel_denoised(k3,threshold,threshold_all,skeleton_method);
new_ker3 = mat2gray(denoised_k3);
new_ker3 = uint8(255*new_ker3);

end

