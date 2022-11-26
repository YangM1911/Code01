function [result] = ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring,im_org)
H = size(y,1);    W = size(y,2);
y_pad = wrap_boundary_liu(y, opt_fft_size([H W]+size(kernel)-1));
im_org = wrap_boundary_liu(im_org, opt_fft_size([H W]+size(kernel)-1));
Latent_tv = [];
% for c = 1:size(y,3)
% 	Latent_tv(:,:,c) = deblurring_adm_aniso(y_pad(:,:,c),kernel, lambda_tv, 1);
% end
% % % B = y_pad;k = kernel;
% % % lambda = lambda_tv;
% % % alpha = 1;
Latent_tv = deblurring_adm_aniso(y_pad,kernel, lambda_tv, 1,im_org);
Latent_tv = Latent_tv(1:H, 1:W, :);
if weight_ring==0
    result = Latent_tv;
    return;
end
[Latent_l0] = Copy_of_L0Restoration(y_pad,kernel, lambda_l0, 2,im_org);
Latent_l0 = Latent_l0(1:H, 1:W, :);

%%
diff = Latent_tv - Latent_l0;
bf_diff = bilateral_filter(diff, 3, 0.1);
result = Latent_tv - weight_ring*bf_diff;

