clc; clear; close all;

%% 1. 读取并截断数据 (严格对齐 LSTM 代码)
sheetName = '基坑沉降';
data_file = readmatrix('数据.xls', 'Sheet', sheetName); 
y = data_file(:, 2);
y = y(~isnan(y)); % 剔除可能存在的 NaN 值

% 保持数据为列向量，更适合 AR 模型的矩阵点乘运算
subsidence_data = y; 

% 数据分区：前 60 期用于训练，其余用于测试
numTimeStepsTrain = 60;
train_data = subsidence_data(1:numTimeStepsTrain);
test_data = subsidence_data(numTimeStepsTrain:end); % 包含第60期，作为预测第61期的连接点

%% 2. 参数设置
p = 2;                  % AR 模型的阶数 (可根据你的 PACF 图或 AIC 准则调整)
SearchAgents_no = 30;   % 海鸥种群数量
Max_iter = 100;         % 最大迭代次数
dim = p;                % 优化的维度等于 AR 的阶数
lb = -2 * ones(1, dim); % AR 系数的搜索下界
ub =  2 * ones(1, dim); % AR 系数的搜索上界

%% 3. 定义适应度函数句柄
% 将前 60 期的训练数据传给 ObjFun_AR 计算训练适应度
fobj = @(x) ObjFun_AR(x, train_data, p);

%% 4. 运行 SOA 算法进行优化
disp('正在运行 SOA 海鸥算法寻找最优 AR 系数...');
tic;
[Best_AR_coeffs, Best_score, SOA_curve] = SOA(SearchAgents_no, Max_iter, lb, ub, dim, fobj);
toc;

disp('SOA 优化得到的最佳 AR 系数 [phi_1, phi_2, ..., phi_p]:');
disp(Best_AR_coeffs);

%% 5. 衔接 LSTM 流程：测试集多步闭环预测
% 为了和 LSTM 的 predictAndUpdateState 闭环预测公平对比，
% 这里我们也必须使用上一期的“预测值”来预测“下一期”，而不是用真实值。

numTimeStepsTest = length(test_data) - 1; % 需要预测的期数 (例如后 5 期)
YPred_AR = zeros(numTimeStepsTest, 1);

% 初始化历史状态：取训练集的最后 p 个点作为最初的已知条件
history_vals = train_data(end-p+1 : end); 

for i = 1:numTimeStepsTest
    % 翻转历史数据，与 AR 系数点乘 (模拟 phi_1*y_{t-1} + phi_2*y_{t-2} + ...)
    % history_vals 是列向量，flipud 翻转后转置为行向量，与 Best_AR_coeffs 相乘
    pred_val = sum(Best_AR_coeffs .* flipud(history_vals)'); 
    
    % 保存当期的预测结果
    YPred_AR(i) = pred_val;
    
    % 更新网络状态 (闭环)：移除最旧的一个数据，把刚才预测出来的最新值填入
    history_vals(1:end-1) = history_vals(2:end);
    history_vals(end) = pred_val; 
end

%% 6. 结果评估
% 提取出真正需要对比的测试集真实值 (去除了第 60 期这个起点)
YTest_real = test_data(2:end); 

% 计算均方根误差 (RMSE)
rmse_AR = sqrt(mean((YPred_AR - YTest_real).^2)); 
fprintf('SOA-AR 模型的多步预测完成，测试集 RMSE: %.4f\n', rmse_AR);

%% 7. 可视化对齐 (出图风格与 LSTM 代码完全一致，方便在论文中并列对比)

% 可视化 1-A：全局趋势图 
figure(1) 
plot(train_data(1:end-1)) 
hold on
idx = numTimeStepsTrain : (numTimeStepsTrain + numTimeStepsTest);
% 把第 60 期真实值作为画图起点，和后面的预测值连起来
plot(idx, [train_data(end); YPred_AR], '.-') 
hold off
xlabel("观测期数")
ylabel("沉降量 (mm)")
title("SOA-AR 基坑沉降量多步预测趋势")
legend(["实际观测训练值" "SOA-AR 模型预测值"])

% 可视化 1-B：测试集对比详情与误差分布
figure(2) 
subplot(2,1,1)
plot(YTest_real, 'b-o') % 真实值加圈
hold on
plot(YPred_AR, 'r.-')   % 预测值加点
hold off
legend(["实际观测值" "SOA-AR 预测值"])
ylabel("沉降量 (mm)")
title("SOA-AR 测试集预测细节对比")

subplot(2,1,2)
stem(YPred_AR - YTest_real) 
xlabel("测试期数 (预测步数)")
ylabel("预测误差 (mm)")
title("SOA-AR 测试集 RMSE = " + num2str(rmse_AR))

% 额外出图：SOA 的收敛曲线，证明海鸥算法起作用了
figure(3)
plot(SOA_curve, 'LineWidth', 2, 'Color', '#0072BD');
title('SOA 海鸥算法寻优收敛曲线');
xlabel('迭代次数');
ylabel('训练集均方根误差 (Fitness)');
grid on;