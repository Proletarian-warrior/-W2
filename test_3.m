clear; close all; clc;
global FCOUNT;

%% 1. 定义目标函数和梯度
function [f, g] = freudenstein_roth(x)
    x1 = x(1);
    x2 = x(2);
    
    f1 = -13 + x1 + 5*x2^2 - x2^3 - 2*x2;
    f2 = -29 + x1 + x2^3 + x2^2 - 14*x2;
    
    f = f1^2 + f2^2;
    
    if nargout > 1
        g = zeros(2, 1);
        g(1) = 2*(f1 + f2);
        g(2) = 2*f1*(10*x2 - 3*x2^2 - 2) + 2*f2*(3*x2^2 + 2*x2 - 14);
    end
end

function [f, g] = FR_count(x)
    global FCOUNT;
    [f, g] = freudenstein_roth(x);
    FCOUNT = FCOUNT + 1;
end

%% 2. NAG法（固定步长）
function [x_opt, f_opt, X_history, Grad_norm_history, total_count] = nag_fixed_step(x0, alpha, mu, tol, max_iter)
    global FCOUNT;
    FCOUNT = 0;
    
    x = x0;
    v = zeros(size(x0));
    
    [f, g] = FR_count(x);
    grad_norm = norm(g);
    
    iter = 0;
    X_history = x;
    Grad_norm_history = grad_norm;
    
    fprintf('=== NAG法 (固定步长) ===\n');
    fprintf('参数: α=%.4f, μ=%.4f\n', alpha, mu);
    
    while grad_norm > tol && iter < max_iter
        x_lookahead = x + mu * v;
        [~, g_lookahead] = FR_count(x_lookahead);
        
        v = mu * v - alpha * g_lookahead;
        x = x + v;
        
        [f, g] = FR_count(x);
        grad_norm = norm(g);
        
        iter = iter + 1;
        X_history(:, end+1) = x;
        Grad_norm_history(end+1) = grad_norm;
        
        if mod(iter, 100) == 0
            fprintf('迭代 %5d: x=[%9.6f, %9.6f], ||g||=%.4e, f=%.4e\n', ...
                    iter, x(1), x(2), grad_norm, f);
        end
    end
    
    x_opt = x;
    f_opt = f;
    total_count = FCOUNT;
    
    fprintf('NAG结果: iter=%d, funcCalls=%d, x=[%.6f, %.6f], ||g||=%.6e\n\n', ...
            iter, total_count, x(1), x(2), grad_norm);
end

%% 3. Adam法（固定步长）
function [x_opt, f_opt, X_history, Grad_norm_history, total_count] = adam_fixed_step(x0, alpha, beta1, beta2, epsilon, tol, max_iter)
    global FCOUNT;
    FCOUNT = 0;
    
    x = x0;
    m = zeros(size(x0));
    v = zeros(size(x0));
    t = 0;
    
    [f, g] = FR_count(x);
    grad_norm = norm(g);
    
    iter = 0;
    X_history = x;
    Grad_norm_history = grad_norm;
    
    fprintf('=== Adam法 (固定步长) ===\n');
    fprintf('参数: α=%.4f, β1=%.4f, β2=%.4f, ε=%.1e\n', alpha, beta1, beta2, epsilon);
    
    while grad_norm > tol && iter < max_iter
        t = t + 1;
        
        m = beta1 * m + (1 - beta1) * g;
        v = beta2 * v + (1 - beta2) * (g.^2);
        
        m_hat = m / (1 - beta1^t);
        v_hat = v / (1 - beta2^t);
        
        x = x - alpha * m_hat ./ (sqrt(v_hat) + epsilon);
        
        [f, g] = FR_count(x);
        grad_norm = norm(g);
        
        iter = iter + 1;
        X_history(:, end+1) = x;
        Grad_norm_history(end+1) = grad_norm;
        
        if mod(iter, 100) == 0
            fprintf('迭代 %5d: x=[%9.6f, %9.6f], ||g||=%.4e, f=%.4e\n', ...
                    iter, x(1), x(2), grad_norm, f);
        end
    end
    
    x_opt = x;
    f_opt = f;
    total_count = FCOUNT;
    
    fprintf('Adam结果: iter=%d, funcCalls=%d, x=[%.6f, %.6f], ||g||=%.6e\n\n', ...
            iter, total_count, x(1), x(2), grad_norm);
end

%% 4. 主程序
x0 = [0.5; -2];
tol = 1e-5;
max_iter = 10000;

% NAG参数
alpha_nag = 0.001;
mu_nag = 0.9;

% Adam参数
alpha_adam = 0.2;
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;

% 运行NAG
[x_nag, f_nag, X_nag, Grad_norms_nag, count_nag] = nag_fixed_step(x0, alpha_nag, mu_nag, tol, max_iter);

% 运行Adam
[x_adam, f_adam, X_adam, Grad_norms_adam, count_adam] = adam_fixed_step(x0, alpha_adam, beta1, beta2, epsilon, tol, max_iter);

%% 5. 定义公共绘图范围
all_x1 = [X_nag(1,:), X_adam(1,:)];
all_x2 = [X_nag(2,:), X_adam(2,:)];

known_points_x1 = [x0(1), 5, 11.41, 0.5];
known_points_x2 = [x0(2), 4, -0.8968, -2];

margin_x1 = 0.15 * (max(all_x1) - min(all_x1));
margin_x2 = 0.15 * (max(all_x2) - min(all_x2));

x1_min = min([all_x1, known_points_x1]) - margin_x1;
x1_max = max([all_x1, known_points_x1]) + margin_x1;
x2_min = min([all_x2, known_points_x2]) - margin_x2;
x2_max = max([all_x2, known_points_x2]) + margin_x2;
x1_min = min([x1_min, x0(1), 5, 11.41, 0.5]);
x1_max = max([x1_max, x0(1), 5, 11.41, 0.5]);
x2_min = min([x2_min, x0(2), 4, -0.8968, -2]);
x2_max = max([x2_max, x0(2), 4, -0.8968, -2]);
x1 = linspace(x1_min, x1_max, 200);
x2 = linspace(x2_min, x2_max, 200);
[X1, X2] = meshgrid(x1, x2);
F = zeros(size(X1));

for i = 1:length(x1)
    for j = 1:length(x2)
        F(j,i) = freudenstein_roth([x1(i); x2(j)]);
    end
end

%% 6. 绘图 - 搜索路径
figure('Position', [100, 100, 1400, 600]);

% 子图1: NAG法
subplot(1, 2, 1);
contour(X1, X2, F, 10, 'LineWidth', 0.5); hold on;

hContourLegend1 = plot(NaN, NaN, 'k-', 'LineWidth', 0.5);
hPath1 = plot(X_nag(1,:), X_nag(2,:), 'ro-', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');

if size(X_nag, 2) > 1
    dx = diff(X_nag(1,:));
    dy = diff(X_nag(2,:));
    hQuiver1 = quiver(X_nag(1,1:end-1), X_nag(2,1:end-1), dx, dy, ...
                      0, 'r', 'LineWidth', 1.2, 'AutoScale', 'off');
else
    hQuiver1 = plot(NaN, NaN, 'r');
end

hStart1 = plot(X_nag(1,1), X_nag(2,1), 'gs', 'MarkerFaceColor', 'g', 'MarkerSize', 8);
hEnd1 = plot(X_nag(1,end), X_nag(2,end), 'bs', 'MarkerFaceColor', 'b', 'MarkerSize', 8);
hGlobal1 = plot(5, 4, 'm^', 'MarkerFaceColor', 'm', 'MarkerSize', 8);
hAlt1 = plot(11.41, -0.8968, 'c^', 'MarkerFaceColor', 'c', 'MarkerSize', 8);

legend([hContourLegend1, hPath1, hQuiver1, hStart1, hEnd1, hGlobal1, hAlt1], ...
       {'等高线', '搜索路径', '搜索方向', '起点', '终点', '全局极小值 [5,4]', '局部极小值 [11.41,-0.8968]'}, ...
       'Location', 'best', 'FontSize', 9);

xlabel('x_1', 'FontSize', 11);
ylabel('x_2', 'FontSize', 11);
title('NAG法搜索路径 (固定步长)', 'FontSize', 12);
grid on;
axis equal;
hold off;

% 子图2: Adam法
subplot(1, 2, 2);
contour(X1, X2, F, 10, 'LineWidth', 0.5); hold on;

hContourLegend2 = plot(NaN, NaN, 'k-', 'LineWidth', 0.5);
hPath2 = plot(X_adam(1,:), X_adam(2,:), 'bo-', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');

if size(X_adam, 2) > 1
    dx2 = diff(X_adam(1,:));
    dy2 = diff(X_adam(2,:));
    hQuiver2 = quiver(X_adam(1,1:end-1), X_adam(2,1:end-1), dx2, dy2, ...
                      0, 'b', 'LineWidth', 1.2, 'AutoScale', 'off');
else
    hQuiver2 = plot(NaN, NaN, 'b');
end

hStart2 = plot(X_adam(1,1), X_adam(2,1), 'gs', 'MarkerFaceColor', 'g', 'MarkerSize', 8);
hEnd2 = plot(X_adam(1,end), X_adam(2,end), 'rs', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
hGlobal2 = plot(5, 4, 'm^', 'MarkerFaceColor', 'm', 'MarkerSize', 8);
hAlt2 = plot(11.41, -0.8968, 'c^', 'MarkerFaceColor', 'c', 'MarkerSize', 8);

legend([hContourLegend2, hPath2, hQuiver2, hStart2, hEnd2, hGlobal2, hAlt2], ...
       {'等高线', '搜索路径', '搜索方向', '起点', '终点', '全局极小值 [5,4]', '局部极小值 [11.41,-0.8968]'}, ...
       'Location', 'best', 'FontSize', 9);

xlabel('x_1', 'FontSize', 11);
ylabel('x_2', 'FontSize', 11);
title('Adam法搜索路径 (固定步长)', 'FontSize', 12);
grid on;
axis equal;
hold off;

%% 7. 绘图 - 梯度范数下降曲线
figure('Position', [100, 100, 1400, 600]);

% 子图1: NAG法梯度范数
subplot(1, 2, 1);
semilogy(0:length(Grad_norms_nag)-1, Grad_norms_nag, 'ro-', 'LineWidth', 2);
hold on;
plot([0, length(Grad_norms_nag)-1], [tol, tol], 'k--', 'LineWidth', 1.5);
xlabel('迭代步数 k', 'FontSize', 11);
ylabel('||∇f||', 'FontSize', 11);
title('NAG法梯度范数下降曲线', 'FontSize', 12);
legend('梯度范数', sprintf('终止条件 (ε=%.0e)', tol), 'Location', 'best');
grid on;

% 子图2: Adam法梯度范数
subplot(1, 2, 2);
semilogy(0:length(Grad_norms_adam)-1, Grad_norms_adam, 'bo-', 'LineWidth', 2);
hold on;
plot([0, length(Grad_norms_adam)-1], [tol, tol], 'k--', 'LineWidth', 1.5);
xlabel('迭代步数 k', 'FontSize', 11);
ylabel('||∇f||', 'FontSize', 11);
title('Adam法梯度范数下降曲线', 'FontSize', 12);
legend('梯度范数', sprintf('终止条件 (ε=%.0e)', tol), 'Location', 'best');
grid on;

%% 8. 统计对比
fprintf('\n=== 统计对比 ===\n');
fprintf('%-12s %-8s %-12s %-12s %-12s\n', '方法', '迭代', '函数调用', '最终||g||', 'f(x)');
fprintf('%-12s %-8d %-12d %-12.6e %-12.6e\n', ...
        'NAG', length(Grad_norms_nag)-1, count_nag, Grad_norms_nag(end), f_nag);
fprintf('%-12s %-8d %-12d %-12.6e %-12.6e\n', ...
        'Adam', length(Grad_norms_adam)-1, count_adam, Grad_norms_adam(end), f_adam);

%% 9. 收敛点分析
global_min = [5; 4];
alt_min = [11.41; -0.8968];

nag_final = X_nag(:, end);
adam_final = X_adam(:, end);

fprintf('\n=== 收敛点分析 ===\n');
fprintf('NAG法:\n');
fprintf('  最终点: [%.6f, %.6f]\n', nag_final(1), nag_final(2));
fprintf('  到全局最小值距离: %.6f\n', norm(nag_final - global_min));
fprintf('  到局部最小值距离: %.6f\n', norm(nag_final - alt_min));

fprintf('Adam法:\n');
fprintf('  最终点: [%.6f, %.6f]\n', adam_final(1), adam_final(2));
fprintf('  到全局最小值距离: %.6f\n', norm(adam_final - global_min));
fprintf('  到局部最小值距离: %.6f\n', norm(adam_final - alt_min));