function wave_x=wave_denosie(wave_y,h,parameter_v,parameter_mu)
% wave_y = LR_WNNM_Main(wave_y, Opts);

H_FFT=fft2(h);
HC_FFT= conj(H_FFT);
wave_x=wave_y;
[cA,cH,cV,cD]=dwt2(wave_x,'haar');
[cH_1,cH_2]=size(cH);
[cV_1,cV_2]=size(cV);
[cD_1,cD_2]=size(cD);

 %%% Lagrange因子初始值 b
 b{1}=zeros(cH_1,cH_2); 
 b{2}=zeros(cV_1,cV_2);
 b{3}=zeros(cD_1,cD_2);
  %%%% d=Wu 初始值
 d{1}=cH;
 d{2}=cV;
 d{3}=cD;
   
  ccA=cA;
   for iter=1:50
     
     w_b{1}=d{1}-b{1};
     w_b{2}=d{2}-b{2};
     w_b{3}=d{3}-b{3};
     rec_w_b=idwt2(ccA,w_b{1},w_b{2},w_b{3},'haar');
%      rec_w_b(322,:)=[];
%      rec_w_b(:,482)=[];
     fft_rec_w_b=fft2(rec_w_b);
%      fft_rec_w_b(:,end) = [];
%      fft_rec_w_b = fft_rec_w_b;
  wave_x=real(ifft2((HC_FFT.*fft2(wave_y)+parameter_v*parameter_mu*fft_rec_w_b)./(HC_FFT.*H_FFT-parameter_v*parameter_mu/norm(wave_x,'fro')+parameter_v*parameter_mu)));
  wave_x(wave_x<0)=0;
  wave_x(wave_x>255)=255;
  [ccA,ccH,ccV,ccD] = dwt2(wave_x,'haar');
  d{1}=Do(1/parameter_v, ccH-d{1});
  d{2}=Do(1/parameter_v, ccV-d{2});
  d{3}=Do(1/parameter_v, ccD-d{3});
  b{1}=b{1}+ccH-d{1};
  b{2}=b{2}+ccV-d{2};
  b{3}=b{3}+ccD-d{3}; 

%     parameter_v=1.02* parameter_v
%     parameter_mu=1.02*parameter_mu
  
  end
 
end

% figure(3)
% 
% imagesc(new_x)
% psnr(uint8(new_x),x_org)
% end 
