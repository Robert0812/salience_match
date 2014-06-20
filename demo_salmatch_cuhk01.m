% clc;
% close all force;
% clear model pwmap_gal pwmap_prb pwmap_trn pwmap_tst pwdist CMC par

% function [CMC_SS, CMC_MS] = Main_ssvm_SvS_campus_new(test_trial)
test_trial = 1;
global dataset baseExp TRIAL gridstep patchsize par

par = struct(...
    'dataset',                      'campus', ... %viper, campus
    'baseExp',                      'structsvm', ...
    'method',                       'test10', ... % 'mask_only', 'mask_sa1', 'mask_sa1_sa2', ...
    'TRIAL',                        test_trial, ... % test_trial
    'gridstep',                     4, ...
    'patchsize',                    10, ...
    'Nr',                           100, ...
    'sigma1',                       3.2, ... % VIPeR: 2.8 CAMPUS: 3.2
    'sigma2',                       0, ...
    'msk_thr',                      0.2, ...
    'norm_data',                    1, ...
    'new_feat',                     1, ...
    'new_match',                    1, ...
    'use_mask',                     1, ...
    'use_model',                    1, ...
    'ensemble',                     0, ...
    'alpha',                        [-1, 0, 0.4, 0.7, 0.6, 0],  ... %[-1, 0, 0.4, 0.7, 0.6, 0], ... %
    'L2',                           1, ...
    'swap',                         1 ...
    );

dataset     = par.dataset;
baseExp     = par.baseExp;
TRIAL       = par.TRIAL;
gridstep    = par.gridstep;
patchsize   = par.patchsize;
Nr          = par.Nr;

if par.L2 
    par.nor = 2;
else
    par.nor = 1;
end
nor  = par.nor;

switch par.method
    
    case 'mask_only'
        phiFun = @(x, y, m1, s1, s2, m2, par) (exp(-double(x).^2/par.sigma1^2).*m1).^nor;
        
    case 'test10'
        phiFun = @phiFun_new;
%         phiFun_update = @phiFun_new_update;
end
%
MS = 1; % multi-shot

project_dir = strcat(pwd, '\');
set_paths;
if par.norm_data
    normdata;
end
initialcontext;

%% extract dense features
if par.new_feat
    build_densefeature_general;
end

%% load features
% this may cost quite a while
feat_gal = zeros(dim, ny*nx, nPerson*2);
feat_prb = zeros(dim, ny*nx, nPerson*2);
hwait = waitbar(0, 'Loading data for testing ...');
for i = 1:nPerson
    load([feat_dir, 'feat', num2str(4*(i-1)+1), '.mat']);
    feat_gal(:, :, 2*i-1) = densefeat;
    load([feat_dir, 'feat', num2str(4*(i-1)+2), '.mat']);
    feat_gal(:, :, 2*i) = densefeat;
    load([feat_dir, 'feat', num2str(4*(i-1)+3), '.mat']);
    feat_prb(:, :, 2*i-1) = densefeat;
    load([feat_dir, 'feat', num2str(4*(i-1)+4), '.mat']);
    feat_prb(:, :, 2*i) = densefeat;
    waitbar(i/nPerson, hwait);
end
close(hwait)

%% compute or load patch matching distances
if par.new_match
    for i = 1:nPerson*2
        i
        for j = 1:nPerson*2
            [pwmap_gal{1, j}, pwmap_gal{2, j}, ~] = mutualmap(feat_gal(:, :, i), feat_prb(:, :, j));
            [pwmap_prb{1, j}, pwmap_prb{2, j}, ~] = mutualmap(feat_prb(:, :, i), feat_gal(:, :, j));
        end
        save([pwdist_dir, 'pwmap_gal_', num2str(i), '.mat'], 'pwmap_gal', '-v7.3');
        save([pwdist_dir, 'pwmap_prb_', num2str(i), '.mat'], 'pwmap_prb', '-v7.3');
    end
end

%% load gallery feature and probe feature for Structural SVM training
clear pwmap_prb_cell pwmap_gal_cell pwmap_all;
tsize = length(ptrn);

IdxP = 2*(ptrn-1) + 1;
IdxG = 2*(ptrn-1) + 2;

for i = 1:tsize
    load([pwdist_dir, 'pwmap_prb_', num2str(IdxP(i)), '.mat']);
    load([pwdist_dir, 'pwmap_gal_', num2str(IdxG(i)), '.mat']);
    pwmap_prb_cell(:, i, :) = pwmap_prb(:, IdxG);
    pwmap_gal_cell(:, i, :) = pwmap_gal(:, IdxP);
end

D_cell = squeeze(pwmap_prb_cell(1, :, :));
P_cell = squeeze(pwmap_prb_cell(2, :, :));

if par.use_salience_in_training
dists_gal = cell2mat(reshape(pwmap_gal_cell(1, :, :), 1, 1, tsize, tsize));
dists_prb = cell2mat(reshape(pwmap_prb_cell(1, :, :), 1, 1, tsize, tsize));
rdists_gal = sort(dists_gal, 4);
rdists_prb = sort(dists_prb, 4);
maxdist_gal = rdists_gal(:, :, :, floor(tsize/2));
maxdist_prb = rdists_prb(:, :, :, floor(tsize/2));
lwdist = min(maxdist_gal(:));
updist = max(maxdist_gal(:));
salience_gal = (maxdist_gal-lwdist)./(updist-lwdist);
salience_prb = (maxdist_prb-lwdist)./(updist-lwdist);
   
salience_gal_cell = squeeze(mat2cell(salience_gal, ny, nx, ones(1, tsize)));
salience_prb_cell = squeeze(mat2cell(salience_prb, ny, nx, ones(1, tsize)));
sg_cell = repmat(salience_gal_cell', tsize, 1);
sp_cell = repmat(salience_prb_cell, 1, tsize);

else 
    sp_cell = repmat({[]}, tsize, tsize);
    sg_cell = repmat({[]}, tsize, tsize);
end
    
% impose pose mask
if par.use_mask
    load([salience_dir, 'posemask.mat']);
    mask_prb = squeeze(mat2cell(mask(:, :, 4*(ptrn-1)+3) >= par.msk_thr, ny, nx, ones(1, tsize)));
    mp_cell = repmat(mask_prb, 1, tsize);

    mask_gal = squeeze(mat2cell(mask(:, :, 4*(ptrn-1)+2) >= par.msk_thr, ny, nx, ones(1, tsize)));
    mg_cell = repmat(mask_gal', tsize, 1);
end

% label 
clear phi;
phi = cell(tsize, tsize);
parfor i = 1:numel(D_cell)
    phi{i} = phiFun(D_cell{i}, P_cell{i}, mp_cell{i}, sp_cell{i}, sg_cell{i}, mg_cell{i}, par);
end

param.phi = phi;
trnIds = 1:tsize;
patterns = num2cell(trnIds);
param.patterns = patterns;
param.pos = cellfun(@(x) find(trnIds == x), patterns, 'UniformOutput', false);
param.neg = cellfun(@(x) setdiff(trnIds, x), param.pos, 'UniformOutput', false);
param.labels = cellfun(@(x, y) cat(2, x, y), param.pos, param.neg, 'UniformOutput', false);

param.lossFn = @mylossCB;
param.constraintFn  = @myconstraintCB;
param.featureFn = @myfeatureCB;
param.dimension = numel(param.phi{1});
param.verbose = 1;
args = ' -c 1 -o 2 -v 1 ';
model = svm_struct_learn(args, param);

% show the number of negtive before positive (before and after training) 
cmc_before = evaluate_pwdist(-cellfun(@(x) dot(ones(1, numel(x)), x(:)), param.phi)');
cmc_after = evaluate_pwdist(-cellfun(@(x) dot(model.w, x(:)), param.phi)');
plot(cmc_before, '-bo'); hold on; plot(cmc_after, '-ro'); title('Training statistics');

clear pwmap_prb_cell pwmap_gal_cell pwmap_all;
clear phi param D_cell P_cell mp_cell mg_cell sp_cell sg_cell;

save([result_dir, 'model_trial', num2str(TRIAL), '.mat'], 'model');

 %% testing
if MS
    gsize = length(ptst)*2;
    IdxP = [2*(ptst-1) + 1, 2*(ptst-1) + 2]'; IdxP = IdxP(:);
    IdxG = [2*(ptst-1) + 1, 2*(ptst-1) + 2]'; IdxG = IdxG(:);
else
    gsize = length(ptst);
    IdxP = 2*(ptst-1) + 1;
    IdxG = 2*(ptst-1) + 2;
end

for i = 1:gsize
    load([pwdist_dir, 'pwmap_prb_', num2str(IdxP(i)), '.mat']);
    load([pwdist_dir, 'pwmap_gal_', num2str(IdxG(i)), '.mat']);
    pwmap_prb_cell(:, i, :) = pwmap_prb(:, IdxG);
    pwmap_gal_cell(:, i, :) = pwmap_gal(:, IdxP);
%     i
end

D_cell = squeeze(pwmap_prb_cell(1, :, :));
P_cell = squeeze(pwmap_prb_cell(2, :, :));
% phi = cellfun(phiFun, D_cell, P_cell, 'UniformOutput', false); 

% compute salience
if MS 
    refsize = length(ptst);
    dists_gal = cell2mat(reshape(pwmap_gal_cell(1, :, 1:2:end), 1, 1, gsize, refsize));
    dists_prb = cell2mat(reshape(pwmap_prb_cell(1, :, 2:2:end), 1, 1, gsize, refsize));
else
    refsize = gsize;
    dists_gal = cell2mat(reshape(pwmap_gal_cell(1, :, :), 1, 1, gsize, gsize));
    dists_prb = cell2mat(reshape(pwmap_prb_cell(1, :, :), 1, 1, gsize, gsize));
end

if par.use_salience_in_testing
rdists_gal = sort(dists_gal, 4);
rdists_prb = sort(dists_prb, 4);
maxdist_gal = rdists_gal(:, :, :, floor(refsize/2));
maxdist_prb = rdists_prb(:, :, :, floor(refsize/2));
lwdist = min(maxdist_gal(:));
updist = max(maxdist_gal(:));
salience_gal = (maxdist_gal-lwdist)./(updist-lwdist);
salience_prb = (maxdist_prb-lwdist)./(updist-lwdist);

salience_gal_cell = squeeze(mat2cell(salience_gal, ny, nx, ones(1, gsize)));
salience_prb_cell = squeeze(mat2cell(salience_prb, ny, nx, ones(1, gsize)));
sg_cell = repmat(salience_gal_cell', gsize, 1);
sp_cell = repmat(salience_prb_cell, 1, gsize);
    
else
    
    sp_cell = repmat({[]}, gsize, gsize);
    sg_cell = repmat({[]}, gsize, gsize);
    
end

%%
if par.use_mask
    load([salience_dir, 'posemask_campus.mat']);
    if MS
        prb_idx = [4*(ptst-1)+3, 4*(ptst-1)+4]'; prb_idx = prb_idx(:);
        gal_idx = [4*(ptst-1)+1, 4*(ptst-1)+2]'; gal_idx = gal_idx(:);
    else
        prb_idx = 4*(ptst-1)+3;
        gal_idx = 4*(ptst-1)+2;
    end
    mask_prb = squeeze(mat2cell(mask(:, :, prb_idx) >= par.msk_thr, ny, nx, ones(1, gsize)));
    mp_cell = repmat(mask_prb, 1, gsize);
    mask_gal = squeeze(mat2cell(mask(:, :, gal_idx) >= par.msk_thr, ny, nx, ones(1, gsize)));
    mg_cell = repmat(mask_gal', gsize, 1);
end

pwdist = zeros(gsize, gsize);

i = 1;
phi_tmp = phiFun(D_cell{i}, P_cell{i}, mp_cell{i}, sp_cell{i}, sg_cell{i}, mg_cell{i}, par);
if par.use_model
    w = model.w;
else
    w = ones(1, numel(phi_tmp));
end

parfor i = 1:numel(D_cell)
    phi_tmp = phiFun(D_cell{i}, P_cell{i}, mp_cell{i}, sp_cell{i}, sg_cell{i}, mg_cell{i}, par);
    pwdist(i) = dot(w, phi_tmp(:));
end

if par.ensemble
    
    pwdist_ssvm = pwdist';
    % pwdist_dfeat = slmetric_pw(reshape(feat_gal, [], gsize), reshape(feat_prb, [], gsize), 'eucdist');
    % pwdist_dfeat = cellfun(@(x) dot(ones(1, numel(x)), x(:)), phi)';
    pwdist_dfeat = pwdist';
    
    if strcmp(par.dataset, 'viper')
        load([pwdist_dir, 'MSCRmatch_VIPeR_f1_Exp007.mat']);
        load([pwdist_dir, 'txpatchmatch_VIPeR_f1_Exp007.mat']);
        load([pwdist_dir, 'wHSVmatch_VIPeR_f1_Exp007.mat']);
    elseif strcmp(par.dataset, 'campus')
        load([pwdist_dir, 'MSCRmatch_campus_f1_Exp007.mat']);
        load([pwdist_dir, 'txpatchmatch_campus_f1_Exp007.mat']);
        load([pwdist_dir, 'wHSVmatch_campus_f1_Exp007.mat']);
    else
        error(0);
    end
    
    pwdist_y = final_dist_y(Sg, Sp);
    pwdist_color = final_dist_color(Sg, Sp);
    pwdist_y = pwdist_y./repmat(max(pwdist_y, [], 1), gsize, 1);
    pwdist_color = pwdist_color./repmat(max(pwdist_color, [], 1), gsize, 1);
    pwdist_hist = final_dist_hist(Sg, Sp);
    pwdist_epitext = dist_epitext(Sg, Sp);
    
    pwdist = par.alpha(1).*pwdist_ssvm + par.alpha(2).*pwdist_dfeat + ...
        par.alpha(3).*pwdist_y + par.alpha(4).*pwdist_color + ...
        par.alpha(5).*pwdist_hist + par.alpha(6).*pwdist_epitext;

else
    pwdist = -pwdist';
    
end

save([pwdist_dir, 'pwdist_trial', num2str(par.TRIAL), '.mat'], 'pwdist');

pwdist1 = pwdist(2:2:end, 1:2:end);
CMC_SS = evaluate_pwdist(pwdist1); % curveCMC(CMC);
fprintf('CMC-rank1(Single-Shot):%2.2f%%\n', CMC_SS(1)*100);

if strcmp(par.dataset, 'campus')
    pwdist_cell = mat2cell(pwdist, 2*ones(1, gsize/2), 2*ones(1, gsize/2));
    pwdist2 = cellfun(@(x) min(x(:)), pwdist_cell);
end

% check baseline performance
CMC_MS = evaluate_pwdist(pwdist2); % curveCMC(CMC);
fprintf('CMC-rank1(Multi-Shot):%2.2f%%\n', CMC_MS(1)*100);

clear pwmap_prb_cell pwmap_gal_cell pwmap_all;
clear phi param D_cell P_cell mp_cell mg_cell sp_cell sg_cell; 

close all force;