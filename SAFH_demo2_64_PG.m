 function SAFH_demo2_64()
n_anchorss = [1000];
% for ian = 52: 52
%     param.n_anchors = 100 * ian;
param.n_anchors = 3000;
for sid = 1: 1
    param.sid = 0.1 * (sid -1);

params51 = [1e-5];
for params_index = 1: length(params51)
    param.o1 = params51(params_index);
% fprintf("param.sid:%.4f,param.o1:%.6f\n",param.sid,param.o1)
seed = 100;
rng('default');
rng(seed);
param.seed = seed;

%% parameters setting
% basie information
param.dataname = 'nus-wide-clear'; %flickr-25k
param.method = 'ASFOH'; 

% method parameters
bits_set = [8,16, 32, 64, 128];%,16, 32, 64, 128
% params = [10, 16, 20, 25, 30, 40, 50, 1000, 10000];%, 32, 64, 128];

param.gamma = 34;%1e4;
param.beta = 1e-4;%1.2e-4;
param.eta = 1e-5;%1.5;
param.rho = 1e-5;
param.omega = 1000;
param.xi = 1e-3;
param.maxIter = 2;
param.deltA = 1e-3;
param.lambda_c = 1e-5;
param.ks = 6e-5;
param.alpha = 0.5;
param.c = 16;

%% load dataset
load("nus-wide-clear.mat");%nus-wide-clear
inx = randperm(size(databaseL,1),180000);
dataset.XTest = XTest;
dataset.YTest = YTest;
dataset.XDatabase = XDatabase(inx,:);
dataset.YDatabase = YDatabase(inx,:);
dataset.testL = testL;
dataset.databaseL = databaseL(inx,:);
trainL = dataset.databaseL;

X1 = dataset.XDatabase;%trans n * q
X2 = dataset.YDatabase;%trans n * q
[n,~] = size(X1);
% SP = 0.8*X1*X1' + 0.2*X2*X2';
%% kernelization
n_anchors = param.n_anchors;
[n, ~] = size(dataset.YDatabase);
anchor_image = dataset.XDatabase(randsample(n, n_anchors),:); 
anchor_text = dataset.YDatabase(randsample(n, n_anchors),:);
dataset.XDatabase = RBF_fast(dataset.XDatabase',anchor_image'); dataset.XTest = RBF_fast(dataset.XTest',anchor_image'); 
dataset.YDatabase = RBF_fast(dataset.YDatabase',anchor_text');  dataset.YTest = RBF_fast(dataset.YTest',anchor_text'); 
[~,sizeL] = size(trainL);
L = trainL';%c*n
% Se = ones(1,sizeL);
% Sl = ones(n, sizeL);
% for i = 1:n
%     B = repmat(L(:,i), 1, n);
%     B1 = B*L' - L*L';
%     for j = 1:sizeL
%         Se(1, j) = norm(B1(:,j));
%     for j = 1:n
%         Se(1, j) = norm(L(:,i) - L(:,j));
%     Sl(i,:) = Se;
%     end
% end

D = L*L';
E = (2*L'*(L*L')-ones(n,1)*(ones(1,n)*L'));% + (1 - param.sid) *param.o1*Sl;
% E1 = L'*L;%-ones(n,1)*ones(1,n);
% [E1, P] = mapminmax(E1, 0, 1);
% E = 0.8*E1 +0.2*E1*E1'/n;
% E = 2*E -ones(n,1)*ones(1,n);
% E = E * L';

%rho:1.000000e-05, ks:1.000000e-03, beta:1.200000e-04,  eta:1.500000e+00, gamma:10000, omega:5.000000e-05, bits:64, train_time:3.7387, ImgToTxt:0.7954, TxtToImg:0.8671
params = [1e-5, 1e-3, 1e-1,2, 100,10000];%, 32, 64, 128]; 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 2, 10, 100, 1000, 10000
params1 = [1.5e-4];%,1.8e-4,2.2e-4,2e-4,2.5e-4];%, 32, 64, 128];
params2 = [2.5e-5];%,2.2e-5,1.5e-5,2e-5,1.8e-5];%, 32, 64, 128];
params3 = [23];%,35,50];%,1e-5];%, 32, 64, 128];
params4 = [1000];%,500, 5000 10000];
params41 = [2e-4];%,5e-4,1e-4,5e-5,8e-5];
params5 = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1,5, 10, 100, 1000, 10000];
params6 = [2e-5, 5e-5, 8e-5, 1e-4, 2e-4, 5e-4, 8e-4];
params7 = [0.5];

%% run algorithm
for bit_index = 1: length(bits_set)
    nbits = bits_set(bit_index);
    param.bits = nbits;
for bit_index = 8: 8
    param.c = bit_index*16;
% for bit_index = 1: 20
%     param.gamma = bit_index*1 + 15;
% for params_index = 1: length(params5)
%     param.omega = params5(params_index);
for params_index = 1: length(params1)
    param.beta = params1(params_index);
for params_index = 1: length(params2)
    param.eta = params2(params_index);
for params_index = 1: length(params3)
    param.gamma = params3(params_index);
for params_index = 1: length(params41)
    param.ks = params41(params_index);
% for params_index = 1: length(params7)
%     param.alpha = params7(params_index);
    %entry
%     E = params(params_index)*E1 +(1-params(params_index))*E1*E1'/n;
%     E = 2*E -ones(n,1)*ones(1,n);
% E = E * L';
    SAFH2_64_LG(trainL, param, dataset, params(1), D, E);
end
end
end
end
% end
end
% end
% clear;

end
end
end
end

