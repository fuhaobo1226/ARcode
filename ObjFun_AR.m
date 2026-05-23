function fitness = ObjFun_AR(x, y_diff, p)
    % x: 海鸥当前位置向量（AR模型的p个系数）
    % y_diff: 差分后的训练集序列
    % p: AR模型阶数
    
    N = length(y_diff);
    n = N - p; % 观测对方方程数
    
    % 构建历史观测矩阵 B0 和常数项 L
    L = y_diff(p+1:end);
    B0 = zeros(n, p);
    for j = 1:p
        B0(:, j) = y_diff(p-j+1:N-j);
    end
    
    % 计算当前系数下的残差
    v = B0 * x' - L; % x是行向量，转置为列向量
    
    % 以残差中误差（σ0）作为适应度，目标是让中误差最小
    fitness = sqrt((v' * v) / n);
end