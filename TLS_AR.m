% AR模型三种方法参数估计对比（LS、SVD、顾及设计矩阵误差的TLS）
% 参考文献：姚宜斌. 顾及设计矩阵误差的AR模型新解法
clear;
clc;

%% 1. 读取并截断数据
sheetName = 'FIN';
data = readmatrix('数据.xls', 'Sheet', sheetName);
y = data(:, 5);
y = y(~isnan(y));


%% 2. AR模型参数设置
p = 2;                    % 阶数 t=3
N = length(y);            % 建模数据总长 30
n = N - p;                % 观测方程数 27

fprintf('===== 数据基本信息 =====\n');
fprintf('建模数据量: N = %d, 观测方程数: n = %d, 阶数: p = %d\n\n', N, n, p);

L = y(p+1:end);
B0 = zeros(n, p);
for j = 1:p
    B0(:, j) = y(p-j+1:N-j);
end

%% ================== 方法一：经典最小二乘 (LS) ==================
X_LS = (B0' * B0) \ (B0' * L);
v_LS = B0 * X_LS - L;
% 匹配论文表5口径：分母用 n=27
sigma0_LS = sqrt((v_LS' * v_LS) / n);

fprintf('===== 1. 经典最小二乘 (LS) 结果 =====\n');
fprintf('X_1 = %8.4f, X_2 = %8.4f, X_3 = %8.4f\n', X_LS);
fprintf('中误差 σ0 = %.4f \n\n', sigma0_LS);

%% ================== 方法二：奇异值分解法 (SVD) ==================
C = [B0, L];
[~, ~, V_svd] = svd(C, 'econ');
V_min = V_svd(:, end);          
X_SVD = -V_min(1:p) / V_min(end);
v_SVD = B0 * X_SVD - L;
sigma0_SVD = sqrt((v_SVD' * v_SVD) / n); 

fprintf('===== 2. 奇异值分解法 (SVD) 结果 =====\n');
fprintf('X_1 = %8.4f, X_2 = %8.4f, X_3 = %8.4f\n', X_SVD);
fprintf('中误差 σ0 = %.4f \n\n', sigma0_SVD);

%% ================== 方法三：同步更新观测值的 TLS (迭代中不带权) ==================
fprintf('===== 3. 同步更新观测值的 TLS 迭代解算 =====\n');
X0_upd = X_LS;
y_upd = y; 
max_iter = 100;
tol = 1e-10;
I_n = eye(n);
I_p = eye(p);

for iter = 1:max_iter
    L_curr = y_upd(p+1:end);
    B0_curr = zeros(n, p);
    for j = 1:p
        B0_curr(:, j) = y_upd(p-j+1:N-j);
    end
    
    B10_B20 = zeros(n, p + n);
    phi_rev = flipud(X0_upd)'; 
    for i = 1:n
        B10_B20(i, i:i+p-1) = phi_rev;
    end
    B10 = B10_B20(:, 1:p);      
    B20 = B10_B20(:, p+1:end);  

    inv_E_B20 = (I_n - B20) \ I_n;
    Bg = [zeros(p, p), I_p; inv_E_B20 * B0_curr, inv_E_B20 * B10]; 
    lg = [zeros(p, 1); inv_E_B20 * (L_curr - B0_curr * X0_upd)]; 

    % 迭代逻辑：不带权
    xg = (Bg' * Bg) \ (Bg' * lg);
    
    X0_upd = X0_upd + xg(1:p);
    vg_step = Bg * xg - lg; 
    y_upd = y_upd + vg_step; 
    
    delta_X = norm(xg(1:p)) / norm(X0_upd);
    if delta_X < tol, break; end
end

fprintf('  => 迭代 %d 次收敛\n', iter);
fprintf('  => 参数结果: X = [%.4f, %.4f, %.4f]\n', X0_upd(1), X0_upd(2), X0_upd(3));

%% ================== 仅在精度评定中加入修正权阵 P ==================
% 严格对照论文 Eq(11) 构造权阵 P: 代表观测值在模型中出现的频数
% 对应频数序列: 2, 3, 4, 4, ..., 3, 2, 1
m_diag = ones(N, 1) * (p + 1);
for i = 1:p
    m_diag(i) = i + 1; 
end
m_diag(N) = 1; 
for i = 1:p-1
    m_diag(N-i) = i + 1;
end
P = diag(m_diag);

% 最终总残差向量 (平差后序列 y_upd 与 原始序列 y 之差)
V_final = y_upd - y;

% 计算加权残差平方和 Ω = V' * P * V
Omega_weighted = V_final' * P * V_final;

% 自由度 f = n - t = 27 - 3 = 24
f = n - p;

% 1. 原始加权中误差 (代表权重为1的单次观测误差)
sigma0_raw = sqrt(Omega_weighted / f);

% 2. 归一化中误差 (代表平均使用频次下的观测值误差，用于与LS对比)
mean_P = mean(m_diag);
sigma0_paper = sigma0_raw / sqrt(mean_P);

fprintf('\n===== 最终精度评定  =====\n');
fprintf('加权残差平方和 Ω = %.6f\n', Omega_weighted);
fprintf('平均权重因子 mean(P) = %.4f\n', mean_P);
fprintf('原始加权中误差 σ0_raw = %.4f\n', sigma0_raw);
fprintf('归一化单位权中误差 σ0_paper = %.4f \n', sigma0_paper);