clear; close all; clc;
global FCOUNT;

%% 1. 参数与计数器
FCOUNT = 0;
x0 = [0.5; -2];
tol = 1e-5;

%% 2. 梯度下降法
fprintf('=== 梯度下降法 ===\n');
x = x0;
[f, g] = FR_count(x);
grad_norm = norm(g);
iter = 0;
X_history = x;
Grad_norm_history = grad_norm;

while grad_norm > tol && iter < 1000
    d = -g;
    [alpha, xnew, fnew, ~] = exact_line_search(@(y) FR_count(y), x, d);
    x = xnew;
    [f, g] = FR_count(x);
    grad_norm = norm(g);
    iter = iter + 1;
    X_history(:, end+1) = x;
    Grad_norm_history(end+1) = grad_norm;
    fprintf('迭代 %d: x = [%.6f, %.6f], 梯度范数 = %.6e\n', iter, x(1), x(2), grad_norm);
end

GD_result.X_history = X_history;
GD_result.Grad_norm_history = Grad_norm_history;
GD_result.FCOUNT = FCOUNT;
GD_result.iter = iter;

fprintf('\n梯度下降结果: iter=%d, funcCalls=%d, x=[%.6f, %.6f], ||g||=%.6e\n\n', ...
        iter, FCOUNT, x(1), x(2), grad_norm);

%% 3. 共轭梯度法 (Fletcher-Reeves)
fprintf('=== 共轭梯度法 ===\n');
FCOUNT = 0;
x = x0;
[f, g] = FR_count(x);
grad_norm = norm(g);
d = -g;
iter_cg = 0;
X_history_cg = x;
Grad_norm_history_cg = grad_norm;

while grad_norm > tol && iter_cg < 1000
    g_old = g;
    [alpha, xnew, fnew, ~] = exact_line_search(@(y) FR_count(y), x, d);
    x = xnew;
    [f, g] = FR_count(x);
    grad_norm = norm(g);
    beta = (g' * g) / (g_old' * g_old);
    d = -g + beta * d;
    iter_cg = iter_cg + 1;
    X_history_cg(:, end+1) = x;
    Grad_norm_history_cg(end+1) = grad_norm;
    fprintf('迭代 %d: x = [%.6f, %.6f], 梯度范数 = %.6e\n', iter_cg, x(1), x(2), grad_norm);
end

CG_result.X_history = X_history_cg;
CG_result.Grad_norm_history = Grad_norm_history_cg;
CG_result.FCOUNT = FCOUNT;
CG_result.iter = iter_cg;

fprintf('\n共轭梯度结果: iter=%d, funcCalls=%d, x=[%.6f, %.6f], ||g||=%.6e\n\n', ...
        iter_cg, FCOUNT, x(1), x(2), grad_norm);

%% 4. 绘图范围与等高线
all_x1 = [GD_result.X_history(1,:), CG_result.X_history(1,:)];
all_x2 = [GD_result.X_history(2,:), CG_result.X_history(2,:)];
margin_x1 = 0.15 * (max(all_x1) - min(all_x1));
margin_x2 = 0.15 * (max(all_x2) - min(all_x2));
x1_min = min(all_x1) - margin_x1;
x1_max = max(all_x1) + margin_x1;
x2_min = min(all_x2) - margin_x2;
x2_max = max(all_x2) + margin_x2;
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

%% 5. 绘图（等高线与路径）
figure('Position', [100 100 1400 600]);

subplot(1,2,1);
contour(X1, X2, F, 50, 'LineWidth', 0.5); hold on;
hContourLegend1 = plot(NaN, NaN, 'k-', 'LineWidth', 0.5);
hPath1 = plot(GD_result.X_history(1,:), GD_result.X_history(2,:), 'ro-', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
if size(GD_result.X_history,2) > 1
    dx = diff(GD_result.X_history(1,:)); dy = diff(GD_result.X_history(2,:));
    hQuiver1 = quiver(GD_result.X_history(1,1:end-1), GD_result.X_history(2,1:end-1), dx, dy, 0, 'r', 'LineWidth', 1.2, 'AutoScale', 'off');
else
    hQuiver1 = plot(NaN, NaN, 'r');
end
hStart1 = plot(GD_result.X_history(1,1), GD_result.X_history(2,1), 'gs', 'MarkerFaceColor', 'g');
hEnd1 = plot(GD_result.X_history(1,end), GD_result.X_history(2,end), 'bs', 'MarkerFaceColor', 'b');
hGlobal1 = plot(5, 4, 'm^', 'MarkerFaceColor', 'm');
hAlt1 = plot(11.41, -0.8968, 'c^', 'MarkerFaceColor', 'c');
legend([hContourLegend1, hPath1, hQuiver1, hStart1, hEnd1, hGlobal1, hAlt1], ...
       {'等高线', '搜索路径', '搜索方向', '起点', '终点', '全局极小值 [5,4]', '局部极小值 [11.41,-0.8968]'}, 'Location', 'best');
xlabel('x_1'); ylabel('x_2'); title('梯度下降法搜索路径'); grid on; hold off;

subplot(1,2,2);
contour(X1, X2, F, 50, 'LineWidth', 0.5); hold on;
hContourLegend2 = plot(NaN, NaN, 'k-', 'LineWidth', 0.5);
hPath2 = plot(CG_result.X_history(1,:), CG_result.X_history(2,:), 'bo-', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
if size(CG_result.X_history,2) > 1
    dx2 = diff(CG_result.X_history(1,:)); dy2 = diff(CG_result.X_history(2,:));
    hQuiver2 = quiver(CG_result.X_history(1,1:end-1), CG_result.X_history(2,1:end-1), dx2, dy2, 0, 'b', 'LineWidth', 1.2, 'AutoScale', 'off');
else
    hQuiver2 = plot(NaN, NaN, 'b');
end
hStart2 = plot(CG_result.X_history(1,1), CG_result.X_history(2,1), 'gs', 'MarkerFaceColor', 'g');
hEnd2 = plot(CG_result.X_history(1,end), CG_result.X_history(2,end), 'rs', 'MarkerFaceColor', 'r');
hGlobal2 = plot(5, 4, 'm^', 'MarkerFaceColor', 'm');
hAlt2 = plot(11.41, -0.8968, 'c^', 'MarkerFaceColor', 'c');
legend([hContourLegend2, hPath2, hQuiver2, hStart2, hEnd2, hGlobal2, hAlt2], ...
       {'等高线', '搜索路径', '搜索方向', '起点', '终点', '全局极小值 [5,4]', '局部极小值 [11.41,-0.8968]'}, 'Location', 'best');
xlabel('x_1'); ylabel('x_2'); title('共轭梯度法搜索路径'); grid on; hold off;

%% 6. 梯度范数曲线
figure('Position', [100 100 1400 600]);
subplot(1,2,1);
semilogy(0:GD_result.iter, GD_result.Grad_norm_history, 'ro-', 'LineWidth', 2);
xlabel('迭代步数 k'); ylabel('||∇f||'); title('梯度下降');
grid on;
subplot(1,2,2);
semilogy(0:CG_result.iter, CG_result.Grad_norm_history, 'bo-', 'LineWidth', 2);
xlabel('迭代步数 k'); ylabel('||∇f||'); title('共轭梯度');
grid on;

%% 7. 统计与收敛点分析
fprintf('=== 统计对比 ===\n');
fprintf('%-12s %-8s %-12s %-12s\n', '方法', '迭代', '函数调用', '||g||');
fprintf('%-12s %-8d %-12d %-12.6e\n', '梯度下降', GD_result.iter, GD_result.FCOUNT, GD_result.Grad_norm_history(end));
fprintf('%-12s %-8d %-12d %-12.6e\n', '共轭梯度', CG_result.iter, CG_result.FCOUNT, CG_result.Grad_norm_history(end));

global_min = [5; 4];
alt_min = [11.41; -0.8968];
gd_final = GD_result.X_history(:, end);
cg_final = CG_result.X_history(:, end);
fprintf('\n收敛分析:\n');
fprintf('GD 到全局最小值距离: %.6f, 到局部最小值距离: %.6f\n', norm(gd_final-global_min), norm(gd_final-alt_min));
fprintf('CG 到全局最小值距离: %.6f, 到局部最小值距离: %.6f\n', norm(cg_final-global_min), norm(cg_final-alt_min));

%% 本地函数
function [f, g] = freudenstein_roth(x)
    x1 = x(1); x2 = x(2);
    u = x1 - x2^3 + 5*x2^2 - 2*x2 - 13;
    v = x1 + x2^3 + x2^2 - 14*x2 - 29;
    f = u^2 + v^2;
    if nargout > 1
        g = [2*(u+v); 2*u*(-3*x2^2+10*x2-2) + 2*v*(3*x2^2+2*x2-14)];
    end
end

function [f, g] = FR_count(x)
    global FCOUNT;
    [f, g] = freudenstein_roth(x);
    FCOUNT = FCOUNT + 1;
end

function [amin, xmin, fmin, internal_count] = exact_line_search(fun, x, d)
    x = x(:); d = d(:);
    a0 = 0; step = 0.1;
    f0 = fun(x + a0*d);
    a1 = a0 + step; f1 = fun(x + a1*d);
    if f1 < f0
        while true
            a2 = a1 + step; f2 = fun(x + a2*d);
            if f2 > f1, break; end
            a0 = a1; a1 = a2; f1 = f2; step = step*2;
        end
        lb = a0; ub = a2;
    else
        step = -step; a2 = a1; f2 = f1; a1 = a0 + step; f1 = fun(x + a1*d);
        if f1 < f0
            while true
                a2 = a1 + step; f2 = fun(x + a2*d);
                if f2 > f1, break; end
                a0 = a1; a1 = a2; f1 = f2; step = step*2;
            end
            lb = a2; ub = a0;
        else
            lb = min(a0, a2); ub = max(a0, a2);
        end
    end
    if lb > ub, tmp = lb; lb = ub; ub = tmp; end
    lb = lb - 0.1 * abs(lb); ub = ub + 0.1 * abs(ub);
    phi = @(a) fun(x + a*d);
    [amin, fmin, ~, output] = fminbnd(phi, lb, ub);
    xmin = x + amin * d;
    if isfield(output,'funcCount'), internal_count = output.funcCount; else internal_count = NaN; end

end
