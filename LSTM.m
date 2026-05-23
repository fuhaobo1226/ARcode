clear;
clc;

%% 1. 读取并截断数据
sheetName = '基坑沉降';
data_file = readmatrix('数据.xls', 'Sheet', sheetName); 
y = data_file(:, 2);
y = y(~isnan(y)); 
% 关键一步：数据转为行向量 
data = y'; 

%% 2. 衔接 PDF 流程：数据分区与标准化
% 将序列的前 60期 用于训练，后 5期 用于测试
numTimeStepsTrain = 60;
dataTrain = data(1:numTimeStepsTrain);
dataTest = data(numTimeStepsTrain:end);

% 标准化数据以防止训练发散 
mu = mean(dataTrain); 
sig = std(dataTrain); 
dataTrainStandardized = (dataTrain - mu) / sig;

% 准备预测变量 (XTrain) 和响应 (YTrain)
% 响应是移位了一个时间步的训练序列
XTrain = dataTrainStandardized(1:end-1); 
YTrain = dataTrainStandardized(2:end);   

%% 3. 衔接 PDF 流程：定义网络架构与选项
% 创建 200 个隐含单元的 LSTM 回归网络
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)      
    lstmLayer(numHiddenUnits)            
    fullyConnectedLayer(numResponses)     
    regressionLayer];                    

% 指定训练选项 (Adam 求解器, 250 轮训练)
options = trainingOptions('adam', ...
    'MaxEpochs', 250, ...
    'GradientThreshold', 1, ...          % 防止梯度爆炸
    'InitialLearnRate', 0.005, ...       
    'LearnRateSchedule', 'piecewise', ... 
    'LearnRateDropPeriod', 125, ...      
    'LearnRateDropFactor', 0.2, ...      
    'Verbose', 0, ...
    'Plots', 'training-progress');       % 显示训练进度图

% 训练 LSTM 网络
net = trainNetwork(XTrain, YTrain, layers, options);

%% 4. 衔接 PDF 流程：预测将来时间步 (闭环预测)
% 使用与训练数据相同的参数标准化测试数据
dataTestStandardized = (dataTest - mu) / sig; 
XTest = dataTestStandardized(1:end-1);        
YTest = dataTest(2:end); % 提取真实的测试集用于最后计算误差

% 初始化网络状态：先对训练数据进行预测，再用 YTrain(end) 开始第一步预测
net = predictAndUpdateState(net, XTrain);
[net, YPred] = predictAndUpdateState(net, YTrain(end));

% 循环其余预测，将前一次预测作为输入 (闭环)
numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net, YPred(:,i)] = predictAndUpdateState(net, YPred(:,i-1), 'ExecutionEnvironment', 'cpu'); 
end

% 使用先前计算的参数对预测进行去标准化
YPred = sig * YPred + mu;

%% 5. 结果评估 (PDF 后续逻辑)
YTest_real = dataTest(2:end); 

% 计算均方根误差 (RMSE)
rmse = sqrt(mean((YPred - YTest_real).^2)); 
fprintf('提取数据的预测完成，RMSE: %.4f\n', rmse);

%% 可视化 1-A：全局趋势图 
figure(1) % 明确指定新建编号为 1 的图窗
plot(dataTrain(1:end-1)) 
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx, [data(numTimeStepsTrain) YPred], '.-')
hold off
xlabel("观测期数")
ylabel("沉降量 (mm)")
title("基坑沉降量多步预测趋势")
legend(["实际观测值" "模型预测值"])

%% 可视化 1-B：测试集对比详情与误差分布
figure(2) % 明确指定新建编号为 2 的图窗
subplot(2,1,1)
plot(YTest) 
hold on
plot(YPred,'.-') 
hold off
legend(["实际观测值" "模型预测值"])
ylabel("沉降量 (mm)")
title("预测细节对比 ")

subplot(2,1,2)
stem(YPred - YTest) 
xlabel("观测期数")
ylabel("预测误差 (mm)")
title("RMSE = " + num2str(rmse))