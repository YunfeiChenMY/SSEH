function [B] = SAFH2_64_LG(trainLabel, param, dataset, SP, D, E)



rng('default');
rng(param.seed);

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
ks = param.ks;
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
c = param.c;
c = size(trainLabel,2);
% X1 = dataset.XDatabase';% q1 * n
% X2 = dataset.YDatabase';% q2 * n
Theta1 = randn(q1,c);
% [TW1,~,HW1] = svd(Th1, 'econ');
% Theta1 = TW1 * HW1';
Theta2= randn(q2,c);
% [TW2,~,HW2] = svd(Th2, 'econ');
% Theta2 = TW2 * HW2';
TA = randn(c,c);
[TA,~,HA] = svd(TA, 'econ');
A = TA * HA';
% A = 0.5 * eye(k) + 0.5/k * ones(k,1)*ones(1, k);
G = randn(c,n);
% F1 = randn(q1,c);
TF1 = randn(q1,c);
[TF1,~,HF1] = svd(TF1, 'econ');
F1 = TF1 * HF1';
% F2 = randn(q2,c);
TF2 = randn(q2,c);
[TF2,~,HF2] = svd(TF2, 'econ');
F2 = TF2 * HF2';
Q = randn(n,k);
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
H = randn(n,k);

ObjectValue = 0;

ObjectValue1 = 1;
D = D;
E = E;
% D = L*L';
% E = 2*L'*L*L'-ones(n,1)*ones(1,n)*L';
% E = 2*(SP)*L'-ones(n,1)*ones(1,n)*L';
tic;
for epoch = 1:maxIter

        % alpha1, alpha2
    Delta1 = norm(X1-F1*G,'fro')^2;
    Delta2 = norm(X2-F2*G,'fro')^2;

    alpha1 = Delta1^(1/(1-gamma)) / (Delta1^(1/(1-gamma))+Delta2^(1/(1-gamma)));
    alpha2 = Delta2^(1/(1-gamma)) / (Delta2^(1/(1-gamma))+Delta2^(1/(1-gamma)));
%     alpha1 = param.alpha;
%     alpha2 = 1 - alpha1;
%     Q = F/(F'*F + eta*eye(k));
%     F = ((alpha1^gamma)*X1'*X1*Q + (alpha2^gamma)*X2'*X2*Q)/((alpha1^gamma)*Q'*X1'*X1*Q + (alpha2^gamma)*Q'*X2'*X2*Q + eta*eye(k));
%     F1 = G'/(G*G' + eta*eye(c));
%     F2 = G'/(G*G' + eta*eye(c));
%     [FU1,~,FV1] = svd(X1*G','econ');
%     F1 = FU1*FV1';
%     [FU2,~,FV2] = svd(X2*G','econ');
%     F2 = FU2*FV2';
    
    %G    
    G = ((alpha1^gamma)*L*X1'*X1*L'+(alpha2^gamma)*L*X2'*X2*L'+R'*P*D*P'*R + beta*R'*R + eta*eye(c))\((alpha1^gamma)*L*X1'*X1 +(alpha2^gamma)*L*X2'*X2 + ks*R'*P*E' + beta*R'*B);
%     G = (R'*P*D*P'*R + beta*R'*R + eta*eye(c))\((alpha1^gamma)*F1'*X1 +(alpha2^gamma)*F2'*X2 + ks*R'*P*E' + beta*R'*B);
%     G = ((alpha1*gamma)*F1'*X1'*X1*F1 + (alpha2*gamma)*F2'*X2'*X2*F2 + R'*P*D*P'*R + beta*R'*R + eta*eye(c) + eye(c))\((alpha1*gamma)*F1'*X1'*X1 +(alpha2*gamma)*F2'*X2'*X2 + ks*R'*P*E' + beta*R'*B);
%     G = ((alpha1^gamma)*F'*X1'*X1*F + (alpha2^gamma)*F'*X2'*X2*F + R'*P*D*P'*R + beta*R'*R + eta*eye(c) + eye(c))\((alpha1^gamma)*F'*X1'*X1 +(alpha2^gamma)*F'*X2'*X2 + ks*R'*P*E' + beta*R'*B);

% G = 1./(1.0+exp(-1.0*G));
% G = @(G)(G.*(G>=0)+0.*(G<0));
% scale = 0.1;
% alpha = 0.1;
% relu1=@(x)(x.*(x>=0)+0.*(x<0));
% leakyrelu1=@(x,scale)(x.*(x>=0)+scale.*x.*(x<0));
% elu=@(x,alpha)(x.*(x>=0)+alpha.*(exp(x)-1).*(x<0));
% G = elu(G, alpha);
%     G = ((alpha1^gamma)*H'*X1'*X1*H + (alpha2^gamma)*H'*X2'*X2*H + R'*P*D*P'*R + beta*R'*R + eta*eye(k))\((alpha1^gamma)*H'*X1'*X1 + (alpha2^gamma)*H'*X2'*X2 + ks*R'*P*E' + beta*R'*B);
%     G = ((alpha1^gamma)*B*X1'*X1*B' + (alpha2^gamma)*B*X2'*X2*B' + R'*P*D*P'*R + beta*R'*R + eta*eye(k))\((alpha1^gamma)*B*X1'*X1 + (alpha2^gamma)*B*X2'*X2 + ks*R'*P*E' + beta*R'*B);
    
    %H
    
    

    %Theta1, Theta2
%     Theta1 = X1*G'/(G*G');
%     Theta2 = X2*G'/(G*G');
%     [U1,~,V1] = svd(X1*G'*A','econ');
%     Theta1 = U1*V1';
%     [U2,~,V2] = svd(X2*G'*A','econ');
%     Theta2 = U2*V2';
     
    % A    
%     [UA,~,VA] = svd((alpha1^gamma)*Theta1'*X1*G'+ (alpha2^gamma)*Theta2'*X2*G','econ');
%     A = UA*VA';
%     
%     
%     
%     % alpha1, alpha2
%     Delta1 = norm (X1-Theta1*A*G,'fro')^2;
%     Delta2 = norm (X2-Theta2*A*G,'fro')^2;
% 
%     alpha1 = Delta1^(1/(1-gamma)) / (Delta1^(1/(1-gamma))+Delta2^(1/(1-gamma)));
%     alpha2 = Delta2^(1/(1-gamma)) / (Delta1^(1/(1-gamma))+Delta2^(1/(1-gamma)));
    
    % G 
%     G = ((alpha1^gamma)*A'*Theta1'*Theta1*A + (alpha2^gamma)*A'*Theta2'*Theta2*A + R'*P*D*P'*R + beta*R'*R + beta*eye(c) + rho*eye(c) + eta*eye(c))\((alpha1^gamma)*A'*Theta1'*X1 + (alpha2^gamma)*A'*Theta2'*X2 + ks*R'*P*E' + beta*R'*B + beta*P*L + rho * B);
%     G = ((alpha1^gamma)*A'*Theta1'*Theta1*A + (alpha2^gamma)*A'*Theta2'*Theta2*A + R'*P*D*P'*R + beta*R'*R + beta*eye(c) + eta*eye(c))\((alpha1^gamma)*A'*Theta1'*X1 + (alpha2^gamma)*A'*Theta2'*X2 + ks*R'*P*E' + beta*R'*B + beta*P*L);

%     G = ((alpha1^gamma)*A'*Theta1'*Theta1*A + (alpha2^gamma)*A'*Theta2'*Theta2*A + R'*P*D*P'*R + beta*R'*R + eta*eye(c))\(ks*R'*P*E' + beta*R'*B);
%     G = ((alpha1^gamma)*A'*Theta1'*Theta1*A + (alpha2^gamma)*A'*Theta2'*Theta2*A + R'*P*D*P'*R + beta*R'*R + eta*eye(c))\((alpha1^gamma)*A'*Theta1'*X1 + (alpha2^gamma)*A'*Theta2'*X2 + ks*R'*P*E' + beta*R'*H);
        
    % P
%     P = (R*G*G'*R' + (beta+eta)*eye(k))\(ks*R*G*E + beta*G*L')/(D);
    P = (R*G*G'*R' + (beta+eta)*eye(k))\(ks*R*G*E + beta*B*L')/(D);
%     P = (R*G*G'*R' + (beta+eta)*eye(k))\(ks*R*G*E + beta*H*L')/(D);
        
    % R
%     R = sylvester(P*D*P' + beta*eye(k), eta*eye(c)/(G*G'), (ks*P*E'*G' + beta*B*G')*(G*G'));
%     [UAR,~,VAR] = svd(2*ks*P*E'*G'-P*L*L'*P'*U*G*G' + 2*beta*R'*B - beta*U*G*G' + omega*(U-Jr/omega),'econ');
    [UAR,~,VAR] = svd(2*ks*P*E'*G'-P*D*P'*U*G*G' + 2*beta*B*G' - beta*U*G*G' + omega*(U-Jr/omega),'econ');
%     [UAR,~,VAR] = svd(2*ks*P*E'*G'-P*D*P'*U*G*G' + 2*beta*H*G' - beta*U*G*G' + omega*(U-Jr/omega),'econ');
    R = UAR*VAR';
    
    % U
    [UAU,~,VAU] = svd(-P*D*P'*R*G*G' - beta*R*G*G' + omega*R - Jr,'econ');
    U = UAU*VAU';
    
    % Jr
    Jr = Jr + omega*(R - U);
%     omega = 10 * omega;

   %H
%    H = beta*P*L + rho * B;
% H = G'/(G*G');
    
    % B
%      B = sgn(beta*R*G + rho*H);
%     B = sgn(beta*R*G);
%     B = sgn(beta*R*G + rho*G);
    B = sgn(beta*R*G + beta*P*L);
    
    
    
    
%     % B  n * k
%     B = sgn(eta*L*R + omega*H);
%     
%     % R  c * k    
%     R = C\(ks*D*H+eta*L'*B)/(H'*H+(eta+rho)*eye(k));

%     % H
%     % O
%     O = (alpha1^gammA)*X1*(P1') + (alpha2^gammA)*X2*(P2') + ks*D'*R + omega*B;
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
% if mod(epoch, 7) == 0
% Recall = size(B, 2);
% % xi = param.xi;
% 
% xi = param.xi;
% 
% % B = B'
% % W1 = (X1'*X1 + xi*eye(q1))\(X1'*B);% q * k
% W1 = B*X1'/(X1*X1'+ xi*eye(q1)); %k * n
% 
% % S = 2*L'*L-ones(n,1)*(ones(1,n)); ks*B*S'*X1'
% % W1 = (B*B' + eta1 * eye(k))\( 2*ks*B*L'*L*X1'+ eta1*B*X1')/(X1*X1'+ xi*eye(q1));
% tBX = sgn(W1*XTest);% q * k
% sim_it = B' * tBX;% n * q
% ImgToTxt = mAP(sim_it,databaseL,testL,Recall);
% 
% 
% 
% W2 = B*X2'/(X2*X2'+ xi*eye(q2)); %k * n
% % W2 = (B*B' + eta1 * eye(k))\( 2*ks*B*L'*L*X2' - ks*B*ones(n,1)*ones(1,n)*X2' + eta1*B*X2')/(X2*X2'+ xi*eye(q2));
% % W2 = (B*B' + eta1 * eye(k))\( 2*ks*B*L'*L*X2' + eta1*B*X2')/(X2*X2'+ xi*eye(q2));
% tBY = sgn(W2*YTest);% q * k
% sim_ti = B' * tBY;% n * q
% TxtToImg = mAP(sim_ti,databaseL,testL,Recall);
% % 
% % fprintf("rho:%d, ks:%d, beta:%d,  eta:%d, gamma:%d, omega:%d, bits:%d, ImgToTxt:%.4f, TxtToImg:%.4f\n", rho, ks, beta, eta, gamma, omega, k, ImgToTxt, TxtToImg)
% % ObjectValue = (alpha1^gamma)*norm(X1-F1*G,'fro')^2 + (alpha2^gamma)*norm(X2-F2*G,'fro')^2 + norm(ks*2*L'*(L*L')-ones(n,1)*(ones(1,n)*L')-G'*R'*P*(L*L'),'fro')^2+ (beta)*norm(B-R*G,'fro')^2 + (beta)*norm(B-P*L,'fro')^2 + (eta)*norm(G,'fro')^2 + (eta)*norm(P*L,'fro')^2;
% % if epoch ==1
% %     ObjectValue1 = ObjectValue;
% % end
% fprintf("ImgToTxt:%.4f,TxtToImg:%.4f,ObjectValue:%.10f\n",ImgToTxt,TxtToImg,1)% ObjectValue/ObjectValue1)
% end


end

train_time = toc;

%record

Recall = size(B, 2);
% xi = param.xi;

% xi = 0.01;
xi = param.xi;

% B = B'
% W1 = (X1'*X1 + xi*eye(q1))\(X1'*B);% q * k
W1 = B*X1'/(X1*X1'+ xi*eye(q1)); %k * n

% S = 2*L'*L-ones(n,1)*(ones(1,n)); ks*B*S'*X1'
% W1 = (B*B' + eta1 * eye(k))\( 2*ks*B*L'*L*X1'+ eta1*B*X1')/(X1*X1'+ xi*eye(q1));
tBX = sgn(W1*XTest);% q * k
sim_it = B' * tBX;% n * q
ImgToTxt = mAP(sim_it,databaseL,testL,Recall);



W2 = B*X2'/(X2*X2'+ xi*eye(q2)); %k * n
% W2 = (B*B' + eta1 * eye(k))\( 2*ks*B*L'*L*X2' - ks*B*ones(n,1)*ones(1,n)*X2' + eta1*B*X2')/(X2*X2'+ xi*eye(q2));
% W2 = (B*B' + eta1 * eye(k))\( 2*ks*B*L'*L*X2' + eta1*B*X2')/(X2*X2'+ xi*eye(q2));
tBY = sgn(W2*YTest);% q * k
sim_ti = B' * tBY;% n * q
TxtToImg = mAP(sim_ti,databaseL,testL,Recall);

fprintf("time:%s, rho:%d, ks:%d, beta:%d,  eta:%d, gamma:%d, omega:%d, bits:%d, alpha1:%d, c:%d,n_anchors£º%d,param.sid:%.4f,param.o1:%.6f, ImgToTxt:%.4f, TxtToImg:%.4f, A:%.4f\n", train_time,rho, ks, beta, eta, gamma, omega, k, param.alpha, param.c, param.n_anchors,param.sid,param.o1, ImgToTxt, TxtToImg, ImgToTxt+TxtToImg)

end



