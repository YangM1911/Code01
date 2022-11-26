function [R] = multi_scale_fusion(new_ker1,new_ker2,new_ker3)
%%%%%%%%%%%%%%%%%%%%%
lab1 = new_ker1;
lab2 = new_ker2;
lab3 = new_ker3;
R = double(lab1/255);
G = double(lab2/255);
B = double(lab3/255);
% Laplacian score
WL1 = abs(imfilter(R, fspecial('Laplacian'), 'replicate', 'conv')); 
WL2 = abs(imfilter(G, fspecial('Laplacian'), 'replicate', 'conv')); 
WL3 = abs(imfilter(B, fspecial('Laplacian'), 'replicate', 'conv')); 
% Saliency Weighted
WS1 = saliency_detection(lab1);
WS2 = saliency_detection(lab2);
WS3 = saliency_detection(lab3);
% Normalized Weights
[W1, W2, W3] = norm_weight(WL1, WS1, WL2 , WS2, WL3, WS3);
level = 3;
% Gauss pyramid
Weight1 = gaussian_pyramid(W1, level);
Weight2 = gaussian_pyramid(W2, level);
Weight3 = gaussian_pyramid(W3, level);
% laplacian pyramid
r1 = laplacian_pyramid(double(double(lab1)), level);
r2 = laplacian_pyramid(double(double(lab2)), level);
r3 = laplacian_pyramid(double(double(lab3)), level);
% Multi-scale fusion
for i = 1 : level
    R_r{i} = Weight1{i} .* r1{i} + Weight2{i} .* r2{i}+ Weight3{i} .* r3{i};
end
% pyramid reconstruct
R = pyramid_reconstruct(R_r);
% kernel denoising
R(R<0)=0;
R = uint8(R);

end

