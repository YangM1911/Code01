function [I] = deblurring_adm_aniso(B, k, lambda, alpha,im_org)

beta = 1/lambda;
beta_rate = 2*sqrt(2);
%beta_max = 5*2^10;
beta_min = 0.001;
[m n D] = size(B); 
% initialize with input or passed in initialization
I = B; 
% make sure k is a odd-sized   保证模糊核得行数与列数是奇数
if ((mod(size(k, 1), 2) ~= 1) | (mod(size(k, 2), 2) ~= 1))
	fprintf('Error - blur kernel k must be odd-sized.\n');
	return;
end;
[Nomin1, Denom1, Denom2] = computeDenominator(B, k);  %计算分母
Ix = [diff(I, 1, 2), I(:,1,:) - I(:,n,:)];   %diff求差分(导数)；本句是在求第“i”层图像的x_轴方向的梯度（一阶差分）
Iy = [diff(I, 1, 1); I(1,:,:) - I(m,:,:)]; 
% % % % % 小波
% % % % [cA,cH,cV,cD]=dwt2(B,'haar');
% % % % [cH_1,cH_2]=size(cH);
% % % % [cV_1,cV_2]=size(cV);
% % % % [cD_1,cD_2]=size(cD);
% % % % %%% Lagrange因子初始值 b
% % % % b{1}=zeros(cH_1,cH_2); 
% % % % b{2}=zeros(cV_1,cV_2);
% % % % b{3}=zeros(cD_1,cD_2);
% % % % %%%% d=Wu 初始值
% % % % d{1}=cH;
% % % % d{2}=cV;
% % % % d{3}=cD;
% % % %    
% % % % ccA=cA;
%% 小波
[cA,cH,cV,cD]=dwt2(B,'haar');

%%% Lagrange因子初始值 b
b{1}=zeros(size(cH,1),size(cH,2),3); 
b{2}=zeros(size(cV,1),size(cV,2),3);
b{3}=zeros(size(cD,1),size(cD,2),3);

%%%% d=Wu 初始值
d{1}=cH;
d{2}=cV;
d{3}=cD;
   
ccA=cA;
%%
% load Opts;
% lr_lambda = 1.0000e-04;
parameter_v = 0.05;
%% Main loop
Outloop = 1;
while beta > beta_min
%     new_I = LR_WNNM_Main(I, Opts);

    gamma = 1/(2*beta);
    new_gamma = 0.000001 * gamma;
    Denom = Denom1 + gamma*Denom2;   %公式32分母
    %% 小波
     w_b{1}=d{1}-b{1};
     w_b{2}=d{2}-b{2};
     w_b{3}=d{3}-b{3};
     rec_w_b=idwt2(ccA,w_b{1},w_b{2},w_b{3},'haar');
     fft_rec_w_b=fft2(rec_w_b);
    %%
    % subproblem for regularization term
    if alpha==1
        Wx = max(abs(Ix) - beta*lambda, 0).*sign(Ix);  %公式30的解
        Wy = max(abs(Iy) - beta*lambda, 0).*sign(Iy);
    else
        Wx = solve_image(Ix, 1/(beta*lambda), alpha);
        Wy = solve_image(Iy, 1/(beta*lambda), alpha);
    end
      Wxx = [Wx(:,n,:) - Wx(:, 1,:), -diff(Wx,1,2)]; 
      Wxx = Wxx + [Wy(m,:,:) - Wy(1, :,:); -diff(Wy,1,1)]; 
      if size(Nomin1,1) ~= size(fft_rec_w_b,1)
          fft_rec_w_b(size(fft_rec_w_b,1),:,:)=[];  
      end
      if size(Nomin1,2) ~= size(fft_rec_w_b,2)
          fft_rec_w_b(:,size(fft_rec_w_b,2),:)=[];  
      end
      
      Fyout = (Nomin1 + new_gamma*fft_rec_w_b + gamma*fft2(Wxx))./Denom;   %公式32
%       Fyout = (Nomin1 + new_gamma*fft_rec_w_b + gamma*fft2(Wxx) +  fft2(lr_lambda*new_I))./Denom;   %公式32
      I = real(ifft2(Fyout));
      New_I{Outloop} = I;
      All_PSNR(Outloop) = ssim(uint8(255*I),uint8(im_org));
      if Outloop>2
      	  if (All_PSNR(Outloop)-All_PSNR(Outloop-1) < 0)
             
             break;
          end
      end
      
      [ccA,ccH,ccV,ccD] = dwt2(I,'haar');
      d{1} = max(abs(ccH-d{1}) - 1/parameter_v, 0).*sign(ccH-d{1}); 
      d{2} = max(abs(ccV-d{2}) - 1/parameter_v, 0).*sign(ccV-d{2}); 
      d{3} = max(abs(ccD-d{3}) - 1/parameter_v, 0).*sign(ccD-d{3}); 
      b{1}=b{1}+ccH-d{1};
      b{2}=b{2}+ccV-d{2};
      b{3}=b{3}+ccD-d{3}; 
%       for t = 1:3
%           q(:,:,t) = guidedfilter(I(:,:,t), I(:,:,t), 24, 1e-4);
%       end
%       I = q;
      % update the gradient terms with new solution
      Ix = [diff(I, 1, 2), I(:,1,:) - I(:,n,:)]; 
      Iy = [diff(I, 1, 1); I(1,:,:) - I(m,:,:)]; 
    beta = beta/2;
    Outloop = Outloop + 1;
end 
I = New_I{Outloop-1};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Nomin1, Denom1, Denom2] = computeDenominator(y, k)

sizey = size(y);
otfk  = psf2otf(k, sizey); 
Nomin1 = conj(otfk).*fft2(y);
Denom1 = abs(otfk).^2; 
% if higher-order filters are used, they must be added here too
Denom2 = abs(psf2otf([1,-1],sizey)).^2 + abs(psf2otf([1;-1],sizey)).^2;
% Nomin1 = repmat(Nomin1,[1,1,3]);
% Denom1 = repmat(Denom1,[1,1,3]);
% Denom2 = repmat(Denom2,[1,1,3]);



