% ===== 基坑沉降数据平稳性检验与差分自动定阶脚本 =====
clear;
clc;

% 1. 读取数据 (延续你原始代码的读取方式)
sheetName = 'AR边坡';
% 假设你要测的是第8列数据（竖向沉降）
data = readmatrix('数据.xls', 'Sheet', sheetName);
y = data(:,2); 
y = y(~isnan(y)); % 剔除空值，防止报错

% 打印表头
fprintf('\n================ 平稳性与差分定阶分析 =================\n');
fprintf('差分阶数\t p值(ADF检验)\t 标准差(方差)\t 结论判定\n');
fprintf('-------------------------------------------------------\n');

best_d = -1;     % 最佳差分阶数初始值
min_std = inf;   % 最小标准差初始值（设为无穷大）

% 2. 循环测试 0阶(无差分)、1阶差分、2阶差分
for d = 0:2
    
    % 执行差分操作
    if d == 0
        y_test = y;
    else
        y_test = diff(y, d);
    end
    
    % --- 核心计算 ---
    % adftest 函数依赖 MATLAB 的计量经济学工具箱 (Econometrics Toolbox)
    % h=1表示平稳(拒绝原假设)，p_val是具体的概率值
    [h, p_val] = adftest(y_test); 
    
    % 计算当前序列的标准差，用于防止“过度差分”
    current_std = std(y_test);
    
    % 判断文字输出
    if p_val < 0.05
        status_txt = '平稳';
    else
        status_txt = '非平稳';
    end
    
    % 打印这一阶的结果
    fprintf(' %d 阶差分\t %.4f\t\t %.4f\t\t %s\n', d, p_val, current_std, status_txt);
    
    % --- 寻优逻辑：挑选最佳阶数 ---
    % 条件1：必须是平稳的 (p_val < 0.05)
    % 条件2：方差必须比之前记录的更小 (防止过度差分导致人工噪声)
    if p_val < 0.05 && current_std < min_std
        min_std = current_std;
        best_d = d;
    end
end

fprintf('-------------------------------------------------------\n');

% 3. 输出最终结论
if best_d ~= -1
    fprintf('=> 【系统综合判定】：最佳差分阶数为 %d 阶！\n', best_d);
    fprintf('   (理由: 该阶数通过了ADF检验且序列波动方差最小)\n');
else
    fprintf('=> 【警告】：即使2阶差分后依然不平稳(p>=0.05)！\n');
    fprintf('   建议检查数据是否含有强烈的非线性异常，或尝试对数差分。\n');
end
fprintf('=======================================================\n\n');