function [S,PSNR] = Copy_of_L0Restoration(Im,kernel, lambda, kappa,im_org)
if ~exist('kappa','var')
    kappa = 2.0;
end
%% pad image
H = size(Im,1);    W = size(Im,2);
Im = wrap_boundary_liu(Im, opt_fft_size([H W]+size(kernel)-1));
im_org = wrap_boundary_liu(im_org, opt_fft_size([H W]+size(kernel)-1));
%%
S = Im;   %模糊图像
betamax = 1e5;
fx = [1, -1];    %求解图像梯度(算子)
fy = [1; -1];
[N,M,D] = size(Im);
sizeI2D = [N,M];
otfFx = psf2otf(fx,sizeI2D);   %将模糊核转化为光学传递函数（OTF）
otfFy = psf2otf(fy,sizeI2D);
%%
KER = psf2otf(kernel,sizeI2D);
Den_KER = abs(KER).^2;  %公式32分母中“conj(F(K)).*F(K)”
%%
Denormin2 = abs(otfFx).^2 + abs(otfFy ).^2;
if D>1
    Denormin2 = repmat(Denormin2,[1,1,D]);
    KER = repmat(KER,[1,1,D]);
    Den_KER = repmat(Den_KER,[1,1,3]);    %将Den_KER置为三层的   公式32分母中“conj(F(K))*F(K)”
end
Normin1 = conj(KER).*fft2(S);   %conj:求解复数共轭；fft2：二维快速傅里叶变换；公式32分子上“conj(F(K))*F(B)”
%% 
beta = 2*lambda;
%% 小波
[cA,cH,cV,cD]=dwt2(Im,'haar');

%%% Lagrange因子初始值 b
b{1}=zeros(size(cH,1),size(cH,2),3); 
b{2}=zeros(size(cV,1),size(cV,2),3);
b{3}=zeros(size(cD,1),size(cD,2),3);

%%%% d=Wu 初始值
d{1}=cH;
d{2}=cV;
d{3}=cD;
   
ccA=cA;
Outloop = 1;
while beta < betamax
    Denormin   = Den_KER + beta*Denormin2;   %公式32的分母部分，“Denormin2”表示模糊图像两个方向梯度的平方
    %% 小波
    new_beta = 0.00002 * beta;
     w_b{1}=d{1}-b{1};
     w_b{2}=d{2}-b{2};
     w_b{3}=d{3}-b{3};
     rec_w_b=idwt2(ccA,w_b{1},w_b{2},w_b{3},'haar');
     fft_rec_w_b=fft2(rec_w_b);
     %%
     for t = 1:3
          q(:,:,t) = guidedfilter(S(:,:,t), S(:,:,t), 24, 1e-4);
      end
      S = q;
    %%
    h = [diff(S,1,2), S(:,1,:) - S(:,end,:)];   %清晰图像图像梯度_x轴方向（diff(S,1,2)表示：对S的列进行一阶差分）
    v = [diff(S,1,1); S(1,:,:) - S(end,:,:)];
    if D==1
        t = (h.^2+v.^2)<lambda/beta;   %得到小于阈值的像素点矩阵
    else
        t = sum((h.^2+v.^2),3)<lambda/beta;
        t = repmat(t,[1,1,D]);
    end
    h(t)=0; v(t)=0;  %将梯度图中小于阈值的像素点置为0
    Normin2 = [h(:,end,:) - h(:, 1,:), -diff(h,1,2)];
    Normin2 = Normin2 + [v(end,:,:) - v(1, :,:); -diff(v,1,1)];  %公式32分子中的“F(M)”
      if size(Normin1,1) ~= size(fft_rec_w_b,1)
          fft_rec_w_b(size(fft_rec_w_b,1),:,:)=[];  
      end
      if size(Normin1,2) ~= size(fft_rec_w_b,2)
          fft_rec_w_b(:,size(fft_rec_w_b,2),:)=[];  
      end
      
    FS = (Normin1 + new_beta*fft_rec_w_b+beta*fft2(Normin2))./(Denormin+new_beta);
    S = real(ifft2(FS));   %使用逆傅里叶变换并求取实部得到结果
    New_S{Outloop} = S;
	All_PSNR(Outloop) = ssim(uint8(255*S ),uint8(im_org));
	if Outloop > 2
        if (All_PSNR(Outloop)-All_PSNR(Outloop-1) < 0)
            break;
        end
    end
    [ccA,ccH,ccV,ccD] = dwt2(S,'haar');
    t1 = sum((abs(d{1}.^2)),3)<lambda/(beta);
    t1 = repmat(t1,[1,1,D]);
    d{1}(t1) = 0;
    t2 = sum((abs(d{2}.^2 )),3)<lambda/(beta);
    t2 = repmat(t2,[1,1,D]);
    d{2}(t2) = 0;
    t3 = sum((abs(d{3}.^2 )),3)<lambda/(beta);
    t3 = repmat(t3,[1,1,D]);
    d{3}(t3) = 0;

    b{1}=b{1}+ccH-d{1};
    b{2}=b{2}+ccV-d{2};
    b{3}=b{3}+ccD-d{3}; 
    beta = beta*kappa;
    Outloop = Outloop + 1;
end
% figure(2)
% imshow(uint8(255*S))
S = New_S{Outloop-1};
S = S(1:H, 1:W, :);
end
