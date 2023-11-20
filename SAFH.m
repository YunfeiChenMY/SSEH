function [B] = SAFH(trainLabel, param, dataset, SP, D, E)


randn('seed', param.seed);

X1 = dataset.XDatabase;%trans n * q
X2 = dataset.YDatabase;%trans n * q

XTest = dataset.XTest;  % n * q
YTest = dataset.YTest;
testL = dataset.testL;
databaseL = dataset.databaseL;
X1 = X1';%trans n * q.
X2 = X2';%trans n * q

XTest = XTest';  % n * q
YTest = YTest';
% testL = testL';
% databaseL = databaseL';

%params
[q1,~] = size(X1);
[q2,~] = size(X2);
% c = size(trainLabel,2);
c = param.bits;
% c = 128;
n = size(trainLabel,1);
k = param.bits;
maxIter = param.maxIter;
gamma = param.gamma;
beta = param.beta;
eta = param.eta;
% eta1 = param.eta1;
rho = param.rho;
omega = param.omega;
xi = param.xi;
deltA = param.deltA;
lambda_c = param.lambda_c;

%dimo trans
% R = randn (c,k)/10;
% P1 = randn (k,q1)/10;
% P2 = randn (k,q2)/10;
% J1 = randn (k,q1)/10;
% J2 = randn (k,q2)/10;
% H = randn (n,k)/10;
% B = sgn (H);
% Theta1 = zeros (k,q1);
% Theta2 = zeros (k,q2);
% 
% L = trainLabel; % n * c
% 
% C = L' * L; % c * c
alpha1 = 0.5;
alpha2 = 0.5;
% X1 = dataset.XDatabase';% q1 * n
% X2 = dataset.YDatabase';% q2 * n
Th1 = randn(q1,c);
[TW1,~,HW1] = svd(Th1, 'econ');
Theta1 = TW1 * HW1';
Th2= randn(q2,c);
[TW2,~,HW2] = svd(Th2, 'econ');
Theta2 = TW2 * HW2';
TA = randn(c,c);
[TA,~,HA] = svd(TA, 'econ');
A = TA * HA';
% A = 0.5 * eye(k) + 0.5/k * ones(k,1)*ones(1, k);
G = randn(c,n);
% S = randn(n,n);
TU = randn(k,c);
[TU,~,HU] = svd(TU, 'econ');
U = TU * HU';
R = U;
Jr = R - U;
L = trainLabel';
% D = 2*L'*L*L' - L'*ones(n,1)*ones(1,n); % c * n
P = randn(k,size(trainLabel,2));
B = sgn(-1 + (1 -(-1))*rand(k,n));
H = randn(k,n);

ObjectValue = 0;
D = D;
E = E;
% D = L*L';
% E = 2*L'*L*L'-ones(n,1)*ones(1,n)*L';
% E = 2*(SP)*L'-ones(n,1)*ones(1,n)*L';
tic;
for epoch = 1:maxIter

    %Theta1, Theta2
    [U1,~,V1] = svd(X1*G'*A','econ');
    Theta1 = U1*V1';
    [U2,~,V2] = svd(X2*G'*A','econ');
    Theta2 = U2*V2';
     
    % A    
    [UA,~,VA] = svd((alpha1^gamma)*Theta1'*X1*G'+ (alpha2^gamma)*Theta2'*X2*G','econ');
    A = UA*VA';
    
    
    
    % alpha1, alpha2
    Delta1 = norm (X1-Theta1*A*G,'fro')^2;
    Delta2 = norm (X2-Theta2*A*G,'fro')^2;

    alpha1 = Delta1^(1/(1-gamma)) / (Delta1^(1/(1-gamma))+Delta2^(1/(1-gamma)));
    alpha2 = Delta2^(1/(1-gamma)) / (Delta1^(1/(1-gamma))+Delta2^(1/(1-gamma)));
    
    % G 
%     G = ((alpha1^gamma)*A'*Theta1'*Theta1*A + (alpha2^gamma)*A'*Theta2'*Theta2*A + R'*P*D*P'*R + beta*R'*R + beta*eye(c) + rho*eye(c) + eta*eye(c))\((alpha1^gamma)*A'*Theta1'*X1 + (alpha2^gamma)*A'*Theta2'*X2 + k*R'*P*E' + beta*R'*B + beta*P*L + rho * B);
%     G = ((alpha1^gamma)*A'*Theta1'*Theta1*A + (alpha2^gamma)*A'*Theta2'*Theta2*A + R'*P*D*P'*R + beta*R'*R + beta*eye(c) + eta*eye(c))\((alpha1^gamma)*A'*Theta1'*X1 + (alpha2^gamma)*A'*Theta2'*X2 + k*R'*P*E' + beta*R'*B + beta*P*L);

    G = ((alpha1^gamma)*A'*Theta1'*Theta1*A + (alpha2^gamma)*A'*Theta2'*Theta2*A + R'*P*D*P'*R + beta*R'*R + eta*eye(c))\((alpha1^gamma)*A'*Theta1'*X1 + (alpha2^gamma)*A'*Theta2'*X2 + k*R'*P*E' + beta*R'*B);
%     G = ((alpha1^gamma)*A'*Theta1'*Theta1*A + (alpha2^gamma)*A'*Theta2'*Theta2*A + R'*P*D*P'*R + beta*R'*R + eta*eye(c))\((alpha1^gamma)*A'*Theta1'*X1 + (alpha2^gamma)*A'*Theta2'*X2 + k*R'*P*E' + beta*R'*H);
        
    % P
%     P = (R*G*G'*R' + (beta+eta)*eye(k))\(k*R*G*E + beta*G*L')/(D);
    P = (R*G*G'*R' + (beta+eta)*eye(k))\(k*R*G*E + beta*B*L')/(D);
%     P = (R*G*G'*R' + (beta+eta)*eye(k))\(k*R*G*E + beta*H*L')/(D);
        
    % R
%     R = sylvester(P*D*P' + beta*eye(k), eta*eye(c)/(G*G'), (k*P*E'*G' + beta*B*G')*(G*G'));
%     [UAR,~,VAR] = svd(2*k*P*E'*G'-P*L*L'*P'*U*G*G' + 2*beta*R'*B - beta*U*G*G' + omega*(U-Jr/omega),'econ');
    [UAR,~,VAR] = svd(2*k*P*E'*G'-P*D*P'*U*G*G' + 2*beta*B*G' - beta*U*G*G' + omega*(U-Jr/omega),'econ');
%     [UAR,~,VAR] = svd(2*k*P*E'*G'-P*D*P'*U*G*G' + 2*beta*H*G' - beta*U*G*G' + omega*(U-Jr/omega),'econ');
    R = UAR*VAR';
    
    % U
    [UAU,~,VAU] = svd(-P*D*P'*R*G*G' - beta*R*G*G' + omega*R - Jr,'econ');
    U = UAU*VAU';
    
    % Jr
    Jr = Jr + omega*(R - U);
%     omega = 10 * omega;

   %H
%    H = beta*P*L + rho * B;
    
    % B
%      B = sgn(beta*R*G + rho*H);
%     B = sgn(beta*R*G);
%     B = sgn(beta*R*G + rho*G);
    B = sgn(beta*R*G + beta*P*L);
    
    
    
    
%     % B  n * k
%     B = sgn(eta*L*R + omega*H);
%     
%     % R  c * k    
%     R = C\(k*D*H+eta*L'*B)/(H'*H+(eta+rho)*eye(k));

%     % H
%     % O
%     O = (alpha1^gammA)*X1*(P1') + (alpha2^gammA)*X2*(P2') + k*D'*R + omega*B;
%     H = calZ(O);
% 
%     
%     
%     % P
%     P1 = ((2*alpha1^gammA)*(H'*H) + 2 * deltA*eye(k))\((2*alpha1^gammA)*H'*X1 + 2*deltA*J1 - Theta1);
%     P2 = ((2*alpha2^gammA)*(H'*H) + 2 * deltA*eye(k))\((2*alpha2^gammA)*H'*X2 + 2*deltA*J2 - Theta2);
%     
%     % J1,J2
%     [U1,S1,V1] = svd(P1+Theta1./deltA,'econ');
%     a = diag(S1) - lambda_c/deltA;
%     a(a<0)=0;
%     T = diag(a);
%     J1 = U1*T*V1';
%     
%     [U2,S2,V2] = svd(P2+Theta2./deltA,'econ');
%     a = diag(S2) - lambda_c/deltA;
%     a(a<0)=0;
%     T = diag(a);
%     J2 = U2*T*V2';
%     
%     deltA = 1.1*deltA;
%     Theta1 = Theta1 + deltA*(P1-J1);
%     Theta2 = Theta2 + deltA*(P2-J2);

end

train_time = toc;

%record

Recall = size(B, 2);
% xi = param.xi;

xi = 0.01;

% B = B'
% W1 = (X1'*X1 + xi*eye(q1))\(X1'*B);% q * k
W1 = B*X1'/(X1*X1'+ xi*eye(q1)); %k * n

% S = 2*L'*L-ones(n,1)*(ones(1,n)); k*B*S'*X1'
% W1 = (B*B' + eta1 * eye(k))\( 2*k*B*L'*L*X1'+ eta1*B*X1')/(X1*X1'+ xi*eye(q1));
tBX = sgn(W1*XTest);% q * k
sim_it = B' * tBX;% n * q
ImgToTxt = mAP(sim_it,databaseL,testL,Recall);



W2 = B*X2'/(X2*X2'+ xi*eye(q2)); %k * n
% W2 = (B*B' + eta1 * eye(k))\( 2*k*B*L'*L*X2' - k*B*ones(n,1)*ones(1,n)*X2' + eta1*B*X2')/(X2*X2'+ xi*eye(q2));
% W2 = (B*B' + eta1 * eye(k))\( 2*k*B*L'*L*X2' + eta1*B*X2')/(X2*X2'+ xi*eye(q2));
tBY = sgn(W2*YTest);% q * k
sim_ti = B' * tBY;% n * q
TxtToImg = mAP(sim_ti,databaseL,testL,Recall);

fprintf("rho:%d, beta:%d,  eta:%d, gamma:%d, omega:%d, bits:%d, train_time:%.4f, ImgToTxt:%.4f, TxtToImg:%.4f\n", rho,  beta, eta, gamma, omega, k, train_time, ImgToTxt, TxtToImg)

end



