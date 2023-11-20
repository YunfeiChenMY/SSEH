function [B] = FFUH(trainLabel, param, dataset)

X1 = dataset.XDatabase';% q1 * n
X2 = dataset.YDatabase';% q2 * n

% TODO: confirm the dimension
XTest = dataset.XTest'; 
YTest = dataset.YTest';
testL = dataset.testL;
databaseL = dataset.databaseL;



[q1,~] = size(X1);
[q2,~] = size(X2);
n = size(trainLabel,1);
k = param.bits;
percent = param.percent;

if(percent == 0.001)
    c = 1;
elseif(percent == 0.999)
    c = k - 1;
else
    c = percent * k;
end
r = k - c;
c = c / 2;

maxIter = param.maxIter;
mU = param.mu;
gammA = param.gamma;
alphA = param.alpha;
betA = param.beta;
etA =  param.eta;
omegA =  param.omega;
deltA = param.delta;

theta1 = randn(n,n);
% [TW1,~,HW1] = svd(theta1, 'econ');
% theta1 = TW1 * HW1';
theta2 = randn(n,n);
% [TW2,~,HW2] = svd(theta2, 'econ');
% theta2 = TW2 * HW2';

thetac = randn(n,n);
% [TCW1,~,HCW1] = svd(thetac, 'econ');
% thetac = TCW1 * HCW1';
% thetac2 = randn(n,n);

H = randn(k,n);
% H2 = randn(k,n);

R = randn(k,n);
% R2 = randn(k,n);

% V = randn(r,n);
% U1 =  randn (q1,r);
% U2 =  randn (q2,r);
% Rc = randn(c,c);


%W1 = randn(q1,c);
%[PW1,~,QW1] = svd(W1, 'econ');
%W1 = PW1 * QW1';

%W2 = randn(q2,c);
%[PW2,~,QW2] = svd(W2, 'econ');
%W2 = PW2 * QW2';

W1 = randn(n,k);
% [PW1,~,QW1] = svd(W1, 'econ');
% W1 = PW1 * QW1';

W2 = randn(n,k);
% [PW2,~,QW2] = svd(W2, 'econ');
% W2 = PW2 * QW2';

% J1 =  randn (q1,c);
% [PJ1,~,QJ1] = svd(J1, 'econ');
% J1 = PJ1 * QJ1';
% 
% J2 =  randn (q2,c);
% [PJ2,~,QJ2] = svd(J2, 'econ');
% J2 = PJ2 * QJ2';
% 
% K1 = W1 - J1;
% K2 = W2 - J2;

%Bv = sgn(-1 + (1 -(-1))*rand(r,n));
%B1 = sgn(-1 + (1 -(-1))*rand(c,n));
%B2 = B1;
B = sgn(-1 + (1 -(-1))*rand(k,n));
% B2 = B1;

first = 0;
tic;
for epoch = 1:maxIter
    
    % theta
    theta1 = (X1' * X1 +(mU + 1) * eye(n)) \ (X1' * X1 + W1 * H);
    theta2 = (X2' * X2 +(mU + 1) * eye(n)) \ (X2' * X2 + W2 * H);
    
    % thetac
    thetac = (H' * H + R' * B) / (H' * H + 2 * mU * eye(n) + R' * R);
%     thetac2 = H2 * H2'/(H2 * H2' + R2 * R2' - B2 * R2');
    
    % H
%     BT = 2 * eye(n) - thetac - thetac' + 2 * thetac * thetac';
    H = sylvester(2 * W1' * W1 + 2 * W2' * W2, 2 * eye(n) - thetac - thetac' + 2 * thetac * thetac', 2 * W1' * theta1 + 2 * W2' * theta2);
%     H2 = eye(r) / (eye(r) - thetac2 + thetac2 * thetac2');
    
    % B
    B = sgn(R * thetac);
%     B2 = sgn(R2 * thetac2);
    
    % W
    W1 = 2 * theta1 * H' / (H * H' + 2 * mU * eye(k));
    W2 = 2 * theta2 * H' / (H * H' + 2 * mU * eye(k));
    
    % R
    R = B * thetac' / (thetac * thetac') 
%     R2 = B2 * thetac2' / (thetac2 * thetac2') 
    
% 
%     % lamda
%     delta1 = norm(X1 - U1*V, 'fro')^2;
%     delta2 = norm(X2 - U2*V, 'fro')^2;
%     
%     lamda1 = delta1^(1/(1-mU)) / (delta1^(1/(1-mU)) + delta2^(1/(1-mU)));
%     lamda2 = delta2^(1/(1-mU)) / (delta1^(1/(1-mU)) + delta2^(1/(1-mU)));
%     
%     % U
%     U1 = (lamda1^mU * X1 * V')/(lamda1^mU * (V * V') + gammA * eye(r));
%     U2 = (lamda2^mU * X2 * V')/(lamda2^mU * (V * V') + gammA * eye(r));
%     
%     % W
%     C1 = (alphA - betA - etA) * (X1 * X1') * J1 + 2 * betA * X1 * X2' * W2 * Rc' + 2 * etA * X1 *B1' + deltA * J1 - K1;
%     C2 = (alphA - etA) * (X2 * X2') * J2 + 2 * betA * X2 * X1' * W1 * Rc - betA * (X2 * X2') * J2 * (Rc' * Rc) + 2 * etA * X2 * B2' + deltA * J2 - K2;
%     
%     [PC1,~,QC1] = svd(C1,'econ');
%     [PC2,~,QC2] = svd(C2,'econ');
%     
%     W1 = PC1 * QC1';
%     W2 = PC2 * QC2';
%     
%     % V
%     V = (lamda1^mU * (U1' * U1) + (omegA + gammA) * eye(r))\(lamda1^mU * U1' * X1 + omegA * Bv) + (lamda2^mU * (U2' * U2) + (omegA + gammA) * eye(r))\(lamda2^mU * U2' * X2 + omegA * Bv);
%     
%     % R
%     Rc = betA * W1' * X1 * X2' * W2 / (betA * W2' * (X2 * X2') * W2 + gammA * eye(c));
%     
%     % Bm Bv
%     B1 = sgn(etA * W1' * X1);
%     B2 = sgn(etA * W2' * X2);
%     Bv = sgn(omegA * V);
%     
%     % J
%     D1 = (alphA - betA - etA) * (X1 * X1') * W1 + deltA * W1 + K1;
%     D2 = (alphA - etA) * (X2 * X2') * W2 - betA * (X2 * X2') * W2 * (Rc' * Rc) + deltA * W2 + K2; 
%     
%     [PD1, ~, QD1] = svd(D1, 'econ');
%     [PD2, ~, QD2] = svd(D2, 'econ');
%     
%     J1 = PD1 * QD1';
%     J2 = PD2 * QD2';
%     
%     % K    
%     K1 = K1 - deltA * (W1 - J1);
%     K2 = K2 - deltA * (W2 - J2);
    
%     deltA = deltA * 1.5;

end
train_time = toc;

xi_value = [1e-3];
% B
% B = sgn([W1'*X1;W2'*X2;V]); 
% for xi_index = 1 : length(xi_value)
%     xi = xi_value(xi_index);
%     P1 = B * X1' / (X1 * X1' + xi * eye(q1));
%     P2 = B * X2' / (X2 * X2' + xi * eye(q2));
%     tBX = sgn(P1 * XTest);
%     tBY = sgn(P2 * YTest);
%     sim_ti = B' * tBX;% n * q
%     sim_it = B' * tBY;% n * q
%     Recall = 100;
%     ImgToTxt = mAP(sim_ti,databaseL,testL,Recall);
%     TxtToImg = mAP(sim_it,databaseL,testL,Recall);
%     fprintf("bits:%.d, I2T:%.4f, T2I:%.4f,train_time:%.2f \n",k, ImgToTxt, TxtToImg,train_time);
%     fprintf("-----");
% end

% Wq1 = (B * X1') / (X1 * X1' +  gammA * eye(q1))
% Wq2 = (B * X2') / (X2 * X2' +  gammA * eye(q2))
% B = sgn([Wq * X1]); 
for xi_index = 1 : length(xi_value)
    xi = xi_value(xi_index);
    P1 = B * X1' / (X1 * X1' + xi * eye(q1));
    P2 = B * X2' / (X2 * X2' + xi * eye(q2));
    tBX = sgn(P1 * XTest);
    tBY = sgn(P2 * YTest);
    sim_ti = B' * tBX;% n * q
    sim_it = B' * tBY;% n * q
    Recall = 100;
    ImgToTxt = mAP(sim_ti,databaseL,testL,Recall);
    TxtToImg = mAP(sim_it,databaseL,testL,Recall);
    fprintf("bits:%.d, I2T:%.4f, T2I:%.4f,train_time:%.2f \n",k, ImgToTxt, TxtToImg,train_time);
    fprintf("-----");
end


end



