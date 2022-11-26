function [nw1, nw2,nw3] = norm_weight(w1, w2, w4, w5,w7,w8)
K = 2;
delta = 0.1;

nw1 = w1 + w2 ;
nw2 = w4 + w5 ;
nw3 = w7+w8;
w = nw1 + nw2+nw3;

nw1 = (nw1 + delta) ./ (w + K * delta);
nw2 = (nw2 + delta) ./ (w + K * delta);
nw3 = (nw3 + delta) ./ (w + K * delta);
end