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