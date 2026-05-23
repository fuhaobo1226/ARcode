function [Best_pos, Best_score, Convergence_curve] = SOA(SearchAgents_no, Max_iter, lb, ub, dim, fobj)
    % SearchAgents_no: 海鸥数量 (种群大小)
    % Max_iter: 最大迭代次数
    % lb, ub: 搜索空间的下界和上界
    % dim: 维度 (在这里等于 AR 模型的阶数 p)
    % fobj: 目标函数句柄
    
    % 1. 初始化海鸥种群位置
    Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    
    % 记录全局最优
    Best_pos = zeros(1, dim);
    Best_score = inf; % 求极小值，初始设为无穷大
    
    Convergence_curve = zeros(1, Max_iter);
    
    % SOA 螺旋常数设定
    u = 1; 
    v = 1; 
    fc = 2; % 控制参数的初始常数
    
    % 2. 开始主循环迭代
    for t = 1:Max_iter
        % 评估当前每只海鸥的适应度
        for i = 1:SearchAgents_no
            % 边界处理（防止海鸥飞出搜索范围）
            Flag4ub = Positions(i,:) > ub;
            Flag4lb = Positions(i,:) < lb;
            Positions(i,:) = (Positions(i,:) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
            
            % 计算适应度
            fitness = fobj(Positions(i,:));
            
            % 更新全局最优海鸥
            if fitness < Best_score
                Best_score = fitness;
                Best_pos = Positions(i,:);
            end
        end
        
        % 更新控制参数 A (随迭代次数从 fc 线性递减到 0)
        A = fc - (t * (fc / Max_iter));
        
        % 更新每只海鸥的位置
        for i = 1:SearchAgents_no
            % --- 迁徙阶段 (Exploration) ---
            % 计算避免碰撞的附加位置
            Cs = Positions(i,:) * A; 
            
            % 计算向最优位置移动的步长
            rd = rand();
            B = 2 * (A^2) * rd; 
            Ms = B * (Best_pos - Positions(i,:)); 
            
            % 综合移动意图
            Ds = abs(Cs + Ms); 
            
            % --- 攻击阶段 (Exploitation) ---
            % 螺旋飞行的参数
            theta = rand() * 2 * pi; % 角度 [0, 2pi]
            k = rand(); 
            r_radius = u * exp(k * v); % 螺旋半径
            
            % 三维坐标系的螺旋计算
            x = r_radius * cos(theta);
            y = r_radius * sin(theta);
            z = r_radius * theta;
            
            % 位置更新公式
            Positions(i,:) = Ds * x * y * z + Best_pos;
        end
        
        % 记录当前迭代的最优结果
        Convergence_curve(t) = Best_score;
        
        % 可选：在控制台打印进度
        disp(['Iteration ' num2str(t) ': Best RMSE = ' num2str(Best_score)]);
    end
end