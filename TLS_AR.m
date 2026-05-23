% AR模型三种方法参数估计对比（LS、矩估计法、顾及设计矩阵误差的TLS）
% 参考文献：姚宜斌. 顾及设计矩阵误差的AR模型新解法
% 参考文献：陈子江. 基于时间序列AR(P)模型的边坡变形预测与应用
clear;
clc;

%% ================== 1. 全局参数配置区 (随时可改) ==================
sheetName = '基坑沉降'; % 工作表名称
col_idx = 2;           % 【改这里】读取哪一列数据？ (例如 2=纵向, 3=横向, 4=高程)
diff_order = 1;        % 【改这里】差分几阶？ (0=不差分, 1=一阶差分, 2=二阶差分)

train_num_raw = 58;    % 【改这里】原始数据前多少期用于建模？
pred_num = 7;          % 【改这里】往后预测多少期？
p = 2;                 % 【改这里】AR模型阶数 p

%% ================== 2. 数据读取与自动差分处理 ==================
% 1. 读取指定的整列原始数据并剔除空值
data = readmatrix('数据.xls', 'Sheet', sheetName);
y_all_raw = data(:, col_idx);
y_all_raw = y_all_raw(~isnan(y_all_raw));

% 2. 划分出原始数据的训练集和测试集（供后续算误差用）
y_train_raw = y_all_raw(1:train_num_raw);
y_true = y_all_raw(train_num_raw+1 : train_num_raw+pred_num);

% 3. 根据设定的 diff_order 自动进行差分
if diff_order == 0
    y = y_train_raw;
elseif diff_order == 1
    y = diff(y_train_raw);
elseif diff_order == 2
    y = diff(y_train_raw, 2);
else
    error('目前仅支持 0阶、1阶 或 2阶 差分');
end

% 4. 自动计算最终送入模型的序列长度
N = length(y);            % 差分后的有效建模数据量
n = N - p;                % 观测方程数

fprintf('===== 全局参数与数据状态 =====\n');
fprintf('读取第 %d 列, 采用 %d 阶差分\n', col_idx, diff_order);
fprintf('差分后实际建模数据量 N = %d, 观测方程数 n = %d, 模型阶数 p = %d\n\n', N, n, p);

%% ================== 3. AR模型参数估计矩阵求解 ==================
L = y(p+1:end);
B0 = zeros(n, p);
for j = 1:p
    B0(:, j) = y(p-j+1:N-j);
end

%% ================== 方法一：经典最小二乘 (LS) ==================
X_LS = (B0' * B0) \ (B0' * L);
v_LS = B0 * X_LS - L;
sigma0_LS = sqrt((v_LS' * v_LS) / n);

fprintf('===== 1. 经典最小二乘 (LS) 结果 =====\n');
for i = 1:p
    fprintf('X_%d = %8.4f  ', i, X_LS(i));
end
fprintf('\n中误差 σ0 = %.4f \n\n', sigma0_LS);

%% ================== 方法二：矩估计法 (Yule-Walker) ==================
% 依据陈子江论文中的公式计算样本自协方差和自相关函数
gamma = zeros(p+1, 1);
for k = 0:p
    % 计算样本自协方差函数 \hat{\gamma}_k
    sum_val = 0;
    for t = 1:(N-k)
        sum_val = sum_val + y(t) * y(t+k);
    end
    gamma(k+1) = sum_val / N;
end

% 计算样本自相关函数 \hat{\rho}_k
rho = gamma / gamma(1);

% 构造 Yule-Walker 方程 \hat{\rho} = \hat{R} * \hat{\varphi}
rho_vec = rho(2:p+1);       % 对应方程左侧的自相关系数向量
R_mat = toeplitz(rho(1:p)); % 对应 \hat{R} 矩阵

% 计算模型参数估计值 \hat{\varphi} = \hat{R}^{-1} * \hat{\rho}
X_YW = R_mat \ rho_vec;

% 计算残差和中误差（保持与LS相同的评定方式）
v_YW = B0 * X_YW - L;
sigma0_YW = sqrt((v_YW' * v_YW) / n);

fprintf('===== 2. 矩估计法 (Yule-Walker) 结果 =====\n');
for i = 1:p
    fprintf('X_%d = %8.4f  ', i, X_YW(i));
end
fprintf('\n中误差 σ0 = %.4f \n\n', sigma0_YW);

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
fprintf('  => 参数结果: ');
fprintf('%.4f  ', X0_upd(1:p));
fprintf('\n');

%% ================== 4. 多步动态预测 (得出差分预测值) ==================
y_pred_LS_diff = zeros(pred_num, 1);
y_pred_YW_diff = zeros(pred_num, 1);
y_pred_TLS_diff = zeros(pred_num, 1);

hist_LS = y(end-p+1:end);
hist_YW = y(end-p+1:end);
hist_TLS = y_upd(end-p+1:end); % TLS使用迭代去噪后的观测值尾巴

for k = 1:pred_num
    y_pred_LS_diff(k) = flipud(hist_LS)' * X_LS;
    y_pred_YW_diff(k) = flipud(hist_YW)' * X_YW;
    y_pred_TLS_diff(k) = flipud(hist_TLS)' * X0_upd;
    
    hist_LS = [hist_LS(2:end); y_pred_LS_diff(k)];
    hist_YW = [hist_YW(2:end); y_pred_YW_diff(k)];
    hist_TLS = [hist_TLS(2:end); y_pred_TLS_diff(k)];
end

%% ================== 5. 自动逆差分还原 (核心通用逻辑) ==================
% 将三种方法的差分预测结果放入循环，统一还原为原始沉降量
preds_diff = {y_pred_LS_diff, y_pred_YW_diff, y_pred_TLS_diff};
preds_restore = cell(1, 3);

for m = 1:3
    curr_diff = preds_diff{m};
    
    if diff_order == 0
        % 0阶：不还原，直接输出
        preds_restore{m} = curr_diff;
        
    elseif diff_order == 1
        % 1阶：原值 = 最后一期原值 + 一阶差分的累积和
        last_raw = y_train_raw(end);
        preds_restore{m} = last_raw + cumsum(curr_diff);
        
    elseif diff_order == 2
        % 2阶：先还原出一阶差分序列，再还原出原始序列
        last_raw = y_train_raw(end);
        last_diff1 = y_train_raw(end) - y_train_raw(end-1); % 手工算出最后一期的一阶差分
        
        pred_diff1 = last_diff1 + cumsum(curr_diff);        % 还原出未来的的一阶差分
        preds_restore{m} = last_raw + cumsum(pred_diff1);   % 还原出未来的原始值
    end
end

% 提取最终的原始量级预测值
y_pred_LS  = preds_restore{1};
y_pred_YW  = preds_restore{2};
y_pred_TLS = preds_restore{3};

%% ================== 6. 最终误差计算与输出 ==================
err_LS  = y_true - y_pred_LS;
err_YW  = y_true - y_pred_YW;
err_TLS = y_true - y_pred_TLS;

fprintf('\n===== %d期多步预测结果对比 (已还原为原始量级) =====\n', pred_num);
fprintf('期数\t真实值\t\tLS预测\t\tYW预测\t\tTLS预测\n');
for k = 1:pred_num
    fprintf('%d\t%8.4f\t%8.4f\t%8.4f\t%8.4f\n', k, y_true(k), y_pred_LS(k), y_pred_YW(k), y_pred_TLS(k));
end

fprintf('\n===== 预测精度评价 (MAE & RMSE) =====\n');
fprintf('模型\tMAE\t\tRMSE\n');
fprintf('LS\t%.4f\t\t%.4f\n', mean(abs(err_LS)), sqrt(mean(err_LS.^2)));
fprintf('YW\t%.4f\t\t%.4f\n', mean(abs(err_YW)), sqrt(mean(err_YW.^2)));
fprintf('TLS\t%.4f\t\t%.4f\n', mean(abs(err_TLS)), sqrt(mean(err_TLS.^2)));


%% ================== 7. 绘制预报曲线与误差对比图 ==================
% 【修改点】：统一使用 train_num_raw 而不是 train_num
pred_x = (train_num_raw + 1) : (train_num_raw + pred_num);

% ----- 图 4.1：三种解法的预报曲线局部对比图 -----
figure('Name', '三种解法预报对比', 'Color', 'w', 'Position', [100, 100, 800, 500]);

% 【修改点】：历史背景必须用原始数据 y_train_raw 来画，才能和还原后的预测值衔接！
bg_x = (train_num_raw - 19) : train_num_raw;
plot(bg_x, y_train_raw(end-19:end), 'k-', 'LineWidth', 1.5, 'HandleVisibility','off'); hold on;

% 绘制 8 期预测值与真实值
plot(pred_x, y_true, 'k-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'k');
plot(pred_x, y_pred_LS, 'b-^', 'LineWidth', 1.2);
plot(pred_x, y_pred_YW, 'g-s', 'LineWidth', 1.2);
plot(pred_x, y_pred_TLS, 'r-p', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');

grid on;
% 在第 57 期和第 58 期之间画一条垂直虚线
xline(train_num_raw + 0.5, 'k--', '预测分界线', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 1.5);
xlim([bg_x(1), pred_x(end) + 1]); % 动态锁定 X 轴范围

xlabel('观测期数', 'FontSize', 12);
ylabel('原始沉降量 (mm)', 'FontSize', 12);
% 图表标题自适应显示当前的列和差分阶数
title(sprintf('AR模型预报曲线对比 (读取第%d列 / %d阶差分)', col_idx, diff_order), 'FontSize', 14);
legend('真实观测值', 'LS预测值', '矩估计(YW)预测值', 'TLS预测值', 'Location', 'best');

% ----- 图 4.2：三种解法的预测误差分布柱状图 -----
figure('Name', '三种解法误差对比', 'Color', 'w', 'Position', [150, 150, 800, 400]);

% 绘制 8 期的误差对比
b = bar(pred_x, [err_LS, err_YW, err_TLS], 'grouped');
b(1).FaceColor = [0.2 0.6 1.0]; 
b(2).FaceColor = [0.4 0.8 0.4]; 
b(3).FaceColor = [1.0 0.4 0.4]; 

grid on;
xlim([pred_x(1)-1, pred_x(end)+1]);
xlabel(sprintf('预测期数 (%d-%d期)', pred_x(1), pred_x(end)), 'FontSize', 12);
ylabel('预测绝对误差 (mm)', 'FontSize', 12);
title('不同解法多步预测残差对比', 'FontSize', 14);
legend('LS误差', 'YW误差', 'TLS误差', 'Location', 'best');