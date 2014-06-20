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

function cmc = evaluate_pwdist(pwdist)
% evaluate the performance of pwdist
% assume gallery in dim1, query in dim2, and param test in dim3
% 
N = size(pwdist, 3);
gsize = size(pwdist, 1);    
cmc = zeros(gsize, N);
for i = 1:N
    [~, order] = sort(pwdist(:, :, i));
    match = (order == repmat(1:gsize, [gsize, 1]));
    cmc(:, i) = cumsum(sum(match, 2)./gsize);
end