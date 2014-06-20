%
% Created by Rui Zhao, on Sep 20, 2013. 
% This code is release under BSD license, 
% any problem please contact Rui Zhao rzhao@ee.cuhk.edu.hk
%
% Please cite as
% Rui Zhao, Wanli Ouyang, and Xiaogang Wang. 
% Person Re-identification by Salience Matching. 
% In IEEE International Conference of Computer Vision, 2013. 
%

function [distmat1, idxmat1, pwmap] = mutualmap(feat1, feat2)
% compute the symmetric nearest map 
%
% feat1 [dim][total patch num]
% feat2 [dim][total patch num]
% ny    number of patches in y-axis
% nx    number of patches in x-axis
% type distance type
%
global nx;
global ny;

distmat1 = zeros(ny, nx);
% distmat2 = zeros(ny, nx);
idxmat1 = zeros(ny, nx);
% idxmat2 = zeros(ny, nx);
pwmap = zeros(ny, nx, nx);

for r = 1:ny
    index = ((1:nx)-1)*ny + r;
    pv1 = feat1(:, index);
    pv2 = feat2(:, index);
    pwdist = slmetric_pw(pv1, pv2, 'eucdist');
    % forward
    [mindist, ind] = min(pwdist, [], 2);
    distmat1(r, :) = mindist';
    idxmat1(r, :) = ind';
    pwmap(r, :, :) = pwdist;
    % backward
%     [mindist, ind] = min(pwdist, [], 1);
%     distmat2(r, :) = mindist;
%     idxmat2(r, :) = ind;
end
distmat1 = single(distmat1);
idxmat1 = uint8(idxmat1);