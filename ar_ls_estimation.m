% AR模型最小二乘法参数估计（p阶和p+1阶）
clear;
clc;

% 1. 读取数据
sheetName = '大坝';
data = readmatrix('数据.xls', 'Sheet', sheetName);
y = data(:,2);
y = y(~isnan(y));

% 2. 设置AR阶数
p = 4;
N = length(y);
pacf=parcorr(y);
epsilon=2/sqrt(N);
parcorr(y);

%% ===== p阶AR模型 =====
Y_p = y(p+1:end);

X_p = zeros(N-p, p);
for j = 1:p
    X_p(:, j) = y(p-j+1:N-j);
end

phi_p = inv(X_p' * X_p) * X_p' * Y_p;

% 改正数 V_p = X_p * phi_p - Y_p
V_p = X_p * phi_p - Y_p;
% 残差平方和
Omega_p = V_p' * V_p;

%% ===== p+1阶AR模型 =====
p1 = p + 1;
Y_p1 = y(p1+1:end);

X_p1 = zeros(N-p1, p1);
for j = 1:p1
    X_p1(:, j) = y(p1-j+1:N-j);
end

phi_p1 = inv(X_p1' * X_p1) * X_p1' * Y_p1;

% 改正数 V_{p+1} = X_{p+1} * phi_{p+1} - Y_{p+1}
V_p1 = X_p1 * phi_p1 - Y_p1;
% 残差平方和
Omega_p1 = V_p1' * V_p1;

%% F检验（检验p阶是否足够）
n_p = N - p;                            % p阶样本数
n_p1 = N - p1;                          % p+1阶样本数

% F统计量
%df1 = p1 - p;                           % 分子自由度，一般为1
%df2 = n_p1 - p1;                        % 分母自由度
F_value = (Omega_p - Omega_p1) / (Omega_p1 / (N-p-1));

%% 显示结果
disp('===== p阶AR模型 =====');
disp('AR系数 phi_p:');
disp(phi_p');
disp('残差平方和 Omega_p:');
disp(Omega_p);

disp(' ');
disp('===== p+1阶AR模型 =====');
disp('AR系数 phi_{p+1}:');
disp(phi_p1');
disp('残差平方和 Omega_{p+1}:');
disp(Omega_p1);

disp(' ');
disp('===== F检验结果 =====');
disp(['F统计量 = ' num2str(F_value)]);
