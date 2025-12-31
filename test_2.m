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

%% 2. Armijo-Goldstein 线搜索
function [alpha, x_new, f_new, internal_count] = armijo_goldstein_search(fun, x, d, f0, g0)
    global FCOUNT;
    
    c1 = 0.1;
    c2 = 0.9;
    tau = 0.5;
    alpha = 1;
    max_iter = 50;
    
    internal_count = 0;
    
    for iter = 1:max_iter
        x_new = x + alpha * d;
        f_new = fun(x_new);
        internal_count = internal_count + 1;
        
        armijo_cond = f_new <= f0 + c1 * alpha * g0;
        goldstein_cond = f_new >= f0 + c2 * alpha * g0;
        
        if armijo_cond && goldstein_cond
            break;
        elseif armijo_cond
            break;
        else
            alpha = tau * alpha;
        end
    end
    
    if iter == max_iter
        warning('Armijo-Goldstein线搜索达到最大迭代次数');
    end
end

%% 3. 梯度下降法
function [x_opt, f_opt, X_history, Grad_norm_history, total_count] = gradient_descent_armijo(x0, tol, max_iter)
    global FCOUNT;
    FCOUNT = 0;
    
    x = x0;
    [f, g] = FR_count(x);
    grad_norm = norm(g);
    
    iter = 0;
    X_history = x;
    Grad_norm_history = grad_norm;
    
    fprintf('=== 梯度下降法 (Armijo-Goldstein) ===\n');
    
    while grad_norm > tol && iter < max_iter
        d = -g;
        
        [alpha, x_new, f_new, internal_count] = armijo_goldstein_search(@FR_count, x, d, f, g'*d);
        
        x = x_new;
        [f, g] = FR_count(x);
        grad_norm = norm(g);
        
        iter = iter + 1;
        X_history(:, end+1) = x;
        Grad_norm_history(end+1) = grad_norm;
        
        fprintf('迭代 %3d: α=%.4e, x=[%9.6f, %9.6f], ||g||=%.4e, f=%.4e\n', ...
                iter, alpha, x(1), x(2), grad_norm, f);
    end
    
    x_opt = x;
    f_opt = f;
    total_count = FCOUNT;
    
    fprintf('梯度下降结果: iter=%d, funcCalls=%d, x=[%.6f, %.6f], ||g||=%.6e\n\n', ...
            iter, total_count, x(1), x(2), grad_norm);
end

%% 4. BFGS法
function [x_opt, f_opt, X_history, Grad_norm_history, total_count] = bfgs_armijo(x0, tol, max_iter)
    global FCOUNT;
    FCOUNT = 0;
    
    x = x0;
    [f, g] = FR_count(x);
    grad_norm = norm(g);
    
    n = length(x);
    H = eye(n);
    
    iter = 0;
    X_history = x;
    Grad_norm_history = grad_norm;
    
    fprintf('=== BFGS法 (Armijo-Goldstein) ===\n');
    
    while grad_norm > tol && iter < max_iter
        d = -H * g;
        
        [alpha, x_new, f_new, internal_count] = armijo_goldstein_search(@FR_count, x, d, f, g'*d);
        
        s = alpha * d;
        x_old = x;
        g_old = g;
        
        x = x_new;
        [f, g] = FR_count(x);
        grad_norm = norm(g);
        
        y = g - g_old;
        rho = 1 / (y' * s);
        
        if rho > 0
            I = eye(n);
            H = (I - rho * s * y') * H * (I - rho * y * s') + rho * (s * s');
        end
        
        iter = iter + 1;
        X_history(:, end+1) = x;
        Grad_norm_history(end+1) = grad_norm;
        
        fprintf('迭代 %3d: α=%.4e, x=[%9.6f, %9.6f], ||g||=%.4e, f=%.4e\n', ...
                iter, alpha, x(1), x(2), grad_norm, f);
    end
    
    x_opt = x;
    f_opt = f;
    total_count = FCOUNT;
    
    fprintf('BFGS结果: iter=%d, funcCalls=%d, x=[%.6f, %.6f], ||g||=%.6e\n\n', ...
            iter, total_count, x(1), x(2), grad_norm);
end

%% 5. 主程序
x0 = [0.5; -2];
tol = 1e-5;
max_iter = 1000;

[x_gd, f_gd, X_gd, Grad_norms_gd, count_gd] = gradient_descent_armijo(x0, tol, max_iter);
[x_bfgs, f_bfgs, X_bfgs, Grad_norms_bfgs, count_bfgs] = bfgs_armijo(x0, tol, max_iter);

%% 6. 定义公共绘图范围
all_x1 = [X_gd(1,:), X_bfgs(1,:)];
all_x2 = [X_gd(2,:), X_bfgs(2,:)];

known_points_x1 = [x0(1), 5, 11.41, 0.5];
known_points_x2 = [x0(2), 4, -0.8968, -2];

margin_x1 = 0.15 * (max(all_x1) - min(all_x1));
margin_x2 = 0.15 * (max(all_x2) - min(all_x2));

x1_min = min([all_x1, known_points_x1]) - margin_x1;
x1_max = max([all_x1, known_points_x1]) + margin_x1;
x2_min = min([all_x2, known_points_x2]) - margin_x2;
x2_max = max([all_x2, known_points_x2]) + margin_x2;

x1 = linspace(x1_min, x1_max, 200);
x2 = linspace(x2_min, x2_max, 200);
[X1, X2] = meshgrid(x1, x2);
F = zeros(size(X1));

for i = 1:length(x1)
    for j = 1:length(x2)
        F(j,i) = freudenstein_roth([x1(i); x2(j)]);
    end
end

%% 7. 绘图 - 搜索路径
figure('Position', [100, 100, 1400, 600]);

subplot(1, 2, 1);
contour(X1, X2, F, 50, 'LineWidth', 0.5); hold on;

hContourLegend1 = plot(NaN, NaN, 'k-', 'LineWidth', 0.5);
hPath1 = plot(X_gd(1,:), X_gd(2,:), 'ro-', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');

if size(X_gd, 2) > 1
    dx = diff(X_gd(1,:));
    dy = diff(X_gd(2,:));
    hQuiver1 = quiver(X_gd(1,1:end-1), X_gd(2,1:end-1), dx, dy, ...
                      0, 'r', 'LineWidth', 1.2, 'AutoScale', 'off');
else
    hQuiver1 = plot(NaN, NaN, 'r');
end

hStart1 = plot(X_gd(1,1), X_gd(2,1), 'gs', 'MarkerFaceColor', 'g', 'MarkerSize', 8);
hEnd1 = plot(X_gd(1,end), X_gd(2,end), 'bs', 'MarkerFaceColor', 'b', 'MarkerSize', 8);
hGlobal1 = plot(5, 4, 'm^', 'MarkerFaceColor', 'm', 'MarkerSize', 8);
hAlt1 = plot(11.41, -0.8968, 'c^', 'MarkerFaceColor', 'c', 'MarkerSize', 8);

legend([hContourLegend1, hPath1, hQuiver1, hStart1, hEnd1, hGlobal1, hAlt1], ...
       {'等高线', '搜索路径', '搜索方向', '起点', '终点', '全局极小值 [5,4]', '局部极小值 [11.41,-0.8968]'}, ...
       'Location', 'best', 'FontSize', 9);

xlabel('x_1', 'FontSize', 11);
ylabel('x_2', 'FontSize', 11);
title('梯度下降法搜索路径 (Armijo-Goldstein线搜索)', 'FontSize', 12);
grid on;
axis equal;
hold off;

subplot(1, 2, 2);
contour(X1, X2, F, 50, 'LineWidth', 0.5); hold on;

hContourLegend2 = plot(NaN, NaN, 'k-', 'LineWidth', 0.5);
hPath2 = plot(X_bfgs(1,:), X_bfgs(2,:), 'bo-', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');

if size(X_bfgs, 2) > 1
    dx2 = diff(X_bfgs(1,:));
    dy2 = diff(X_bfgs(2,:));
    hQuiver2 = quiver(X_bfgs(1,1:end-1), X_bfgs(2,1:end-1), dx2, dy2, ...
                      0, 'b', 'LineWidth', 1.2, 'AutoScale', 'off');
else
    hQuiver2 = plot(NaN, NaN, 'b');
end

hStart2 = plot(X_bfgs(1,1), X_bfgs(2,1), 'gs', 'MarkerFaceColor', 'g', 'MarkerSize', 8);
hEnd2 = plot(X_bfgs(1,end), X_bfgs(2,end), 'rs', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
hGlobal2 = plot(5, 4, 'm^', 'MarkerFaceColor', 'm', 'MarkerSize', 8);
hAlt2 = plot(11.41, -0.8968, 'c^', 'MarkerFaceColor', 'c', 'MarkerSize', 8);

legend([hContourLegend2, hPath2, hQuiver2, hStart2, hEnd2, hGlobal2, hAlt2], ...
       {'等高线', '搜索路径', '搜索方向', '起点', '终点', '全局极小值 [5,4]', '局部极小值 [11.41,-0.8968]'}, ...
       'Location', 'best', 'FontSize', 9);

xlabel('x_1', 'FontSize', 11);
ylabel('x_2', 'FontSize', 11);
title('BFGS法搜索路径 (Armijo-Goldstein线搜索)', 'FontSize', 12);
grid on;
axis equal;
hold off;

%% 8. 绘图 - 梯度范数下降曲线
figure('Position', [100, 100, 1400, 600]);

subplot(1, 2, 1);
semilogy(0:length(Grad_norms_gd)-1, Grad_norms_gd, 'ro-', 'LineWidth', 2);
hold on;
plot([0, length(Grad_norms_gd)-1], [tol, tol], 'k--', 'LineWidth', 1.5);
xlabel('迭代步数 k', 'FontSize', 11);
ylabel('||∇f||', 'FontSize', 11);
title('梯度下降法梯度范数下降曲线', 'FontSize', 12);
legend('梯度范数', sprintf('终止条件 (ε=%.0e)', tol), 'Location', 'best');
grid on;

subplot(1, 2, 2);
semilogy(0:length(Grad_norms_bfgs)-1, Grad_norms_bfgs, 'bo-', 'LineWidth', 2);
hold on;
plot([0, length(Grad_norms_bfgs)-1], [tol, tol], 'k--', 'LineWidth', 1.5);
xlabel('迭代步数 k', 'FontSize', 11);
ylabel('||∇f||', 'FontSize', 11);
title('BFGS法梯度范数下降曲线', 'FontSize', 12);
legend('梯度范数', sprintf('终止条件 (ε=%.0e)', tol), 'Location', 'best');
grid on;

%% 9. 统计对比
fprintf('\n=== 统计对比 ===\n');
fprintf('%-12s %-8s %-12s %-12s %-12s\n', '方法', '迭代', '函数调用', '最终||g||', 'f(x)');
fprintf('%-12s %-8d %-12d %-12.6e %-12.6e\n', ...
        '梯度下降', length(Grad_norms_gd)-1, count_gd, Grad_norms_gd(end), f_gd);
fprintf('%-12s %-8d %-12d %-12.6e %-12.6e\n', ...
        'BFGS', length(Grad_norms_bfgs)-1, count_bfgs, Grad_norms_bfgs(end), f_bfgs);

%% 10. 收敛点分析
global_min = [5; 4];
alt_min = [11.41; -0.8968];

gd_final = X_gd(:, end);
bfgs_final = X_bfgs(:, end);

fprintf('\n=== 收敛点分析 ===\n');
fprintf('梯度下降法:\n');
fprintf('  最终点: [%.6f, %.6f]\n', gd_final(1), gd_final(2));
fprintf('  到全局最小值距离: %.6f\n', norm(gd_final - global_min));
fprintf('  到局部最小值距离: %.6f\n', norm(gd_final - alt_min));

fprintf('BFGS法:\n');
fprintf('  最终点: [%.6f, %.6f]\n', bfgs_final(1), bfgs_final(2));
fprintf('  到全局最小值距离: %.6f\n', norm(bfgs_final - global_min));
fprintf('  到局部最小值距离: %.6f\n', norm(bfgs_final - alt_min));