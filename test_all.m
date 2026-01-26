%% ========================================================
% MMA算法测试 - 三问题对比（包含新增的非凸可行域问题）
% 功能: 对比MMA与fmincon算法在两杆桁架和非凸可行域问题上的性能
%% ========================================================
clear; close all; clc;

fprintf('=== MMA算法综合测试：两杆桁架 + 非凸可行域 (n=2) ===\n');

%% =========================
% 问题1：两杆桁架 (2变量)
%% =========================
problem1.name = '两杆桁架 (2变量)';
problem1.fun = @two_bar_truss_standard;
problem1.x0 = [1.5; 0.5];
problem1.lb = [0.1; 0.1];
problem1.ub = [2; 1];
problem1.n_var = 2;

%% =========================
% 问题2：非凸可行域问题 (n=2)
%% =========================
fprintf('\n设置非凸可行域问题 (n=2)...\n');
n = 2;

% 生成矩阵 S, P, Q
[S, P, Q] = generate_matrices_nonconvex(n);
fprintf('矩阵S的特征值: [%f, %f]\n', eig(S));
fprintf('矩阵P的特征值: [%f, %f]\n', eig(P));
fprintf('矩阵Q的特征值: [%f, %f]\n', eig(Q));

problem2.name = '非凸可行域 (n=2)';
problem2.fun = @(x) nonconvex_feasible_region(x, S, P, Q, n);
problem2.x0 = 0.5 * ones(n, 1);
problem2.lb = -ones(n, 1);
problem2.ub = ones(n, 1);
problem2.n_var = n;
problem2.S = S;
problem2.P = P;
problem2.Q = Q;

% 收集所有问题
problems = {problem1, problem2};

%% =========================
% MMA参数
%% =========================
mma_options = struct(...
    'max_iter', 100, ...
    'tol', 1e-6, ...
    's0', 0.30, ...
    'display', true, ...
    'rho0', 1.0, ...      % 降低初始惩罚因子
    'rho_max', 1e6);      % 限制最大惩罚因子

%% =========================
% fmincon参数
%% =========================
fmincon_algorithms = {'interior-point', 'sqp', 'active-set'};
fmincon_options = optimoptions('fmincon', 'Display', 'off', ...
    'MaxIterations', 200, 'OptimalityTolerance', 1e-10, ...
    'ConstraintTolerance', 1e-8);

%% =========================
% 存储所有结果
all_results = struct();

%% =========================
% 循环测试每个问题
for problem_idx = 1:length(problems)
    problem = problems{problem_idx};
    fprintf('\n\n===== 测试问题 %d: %s =====\n', problem_idx, problem.name);
    %% ==== MMA算法 ====
    fprintf('运行MMA算法...\n');
    tic;
    [x_opt_mma, f_opt_mma, history_mma] = MMA_algorithm(...
        problem.fun, problem.x0, problem.lb, problem.ub, mma_options);
    time_mma = toc;
    
    [~, c_mma, ~, ~] = problem.fun(x_opt_mma);
    feasibility_mma = max([0; c_mma]);
    
    %% ==== fmincon算法 ====
    fmincon_results = struct();
    for algo_idx = 1:length(fmincon_algorithms)
        algo_name = fmincon_algorithms{algo_idx};
        fprintf('运行fmincon (%s)...\n', algo_name);
        
        options_fmincon = optimoptions(fmincon_options, 'Algorithm', algo_name);
        tic;
        [x_opt_fmincon, f_opt_fmincon, exitflag, output] = fmincon(...
            @(x) fmincon_obj_wrapper(x, problem.fun), problem.x0, [], [], [], [], ...
            problem.lb, problem.ub, @(x) fmincon_con_wrapper(x, problem.fun), options_fmincon);
        time_fmincon = toc;
        
        [~, c_fmincon, ~, ~] = problem.fun(x_opt_fmincon);
        feasibility_fmincon = max([0; c_fmincon]);
        
        fmincon_results(algo_idx).algorithm = algo_name;
        fmincon_results(algo_idx).x_opt = x_opt_fmincon;
        fmincon_results(algo_idx).f_opt = f_opt_fmincon;
        fmincon_results(algo_idx).iterations = output.iterations;
        fmincon_results(algo_idx).time = time_fmincon;
        fmincon_results(algo_idx).exitflag = exitflag;
        fmincon_results(algo_idx).feasibility = feasibility_fmincon;
    end
    
    %% 存储该问题的结果
    all_results(problem_idx).problem_name = problem.name;
    all_results(problem_idx).n_var = problem.n_var;
    all_results(problem_idx).mma.x_opt = x_opt_mma;
    all_results(problem_idx).mma.f_opt = f_opt_mma;
    all_results(problem_idx).mma.time = time_mma;
    all_results(problem_idx).mma.iterations = length(history_mma.f)-1;
    all_results(problem_idx).mma.feasibility = feasibility_mma;
    all_results(problem_idx).mma.history = history_mma;
    all_results(problem_idx).fmincon = fmincon_results;
    
    %% 显示该问题的结果
    fprintf('\n--- %s 结果汇总 ---\n', problem.name);
    fprintf('%-15s %-12s %-10s %-10s %-12s\n', ...
        '算法', '最优值', '迭代次数', '时间(s)', '可行性');
    fprintf('--------------------------------------------------------------\n');
    
    fprintf('%-15s %-12.6f %-10d %-10.4f %-12.2e\n', ...
        'MMA', f_opt_mma, length(history_mma.f)-1, time_mma, feasibility_mma);
    
    for a = 1:length(fmincon_results)
        r = fmincon_results(a);
        fprintf('%-15s %-12.6f %-10d %-10.4f %-12.2e\n', ...
            r.algorithm, r.f_opt, r.iterations, r.time, r.feasibility);
    end
    
    %% 可视化当前问题
    if problem.n_var == 2  % 只有2变量问题才可视化
        visualize_problem_comparison(problem, history_mma, fmincon_results, time_mma);
     
        if isfield(history_mma, 'subprob') && ~isempty(history_mma.subprob)
            k = min(5, length(history_mma.subprob));   % 画第5步（或最后一步）
            figure('Name', sprintf('%s - MMA第%d次子问题', problem.name, k));
            plot_mma_subproblem_evolution(problem, history_mma);
        end
        end
end
%% =========================
% 综合性能比较
%% =========================
fprintf('\n\n===== 综合性能比较 =====\n');
create_overall_performance_comparison(all_results);

%% =========================
% 保存结果
%% =========================
save('MMA_test_results.mat', 'all_results');
fprintf('\n所有结果已保存到 MMA_test_results.mat\n');

%% =========================
% 辅助函数：非凸可行域问题
%% =========================

function [S, P, Q] = generate_matrices_nonconvex(n)
    S = zeros(n, n);
    P = zeros(n, n);
    Q = zeros(n, n);
    for i = 1:n
        for j = 1:n
            alpha_ij = (i + j - 2) / (2 * n - 2);
            S(i,j) = (2 + sin(4*pi*alpha_ij)) / ((1+abs(i-j))*log(n));
            P(i,j) = (1 + 2*alpha_ij) / ((1+abs(i-j))*log(n));
            Q(i,j) = (3 - 2*alpha_ij) / ((1+abs(i-j))*log(n));
        end
    end
    % 对称矩阵
    S = (S+S')/2;
    P = (P+P')/2;
    Q = (Q+Q')/2;
end

function [f, c, df, dc] = nonconvex_feasible_region(x, S, P, Q, n)
    % 目标函数
    f = x' * S * x;
    % 约束 c <= 0
    c = [n/2 - x'*P*x;
         n/2 - x'*Q*x];
    % 梯度
    if nargout > 2
        df = 2 * S * x;
        dc = [-2*P*x, -2*Q*x];  % 每列对应一个约束
    end
end

%% =========================
% 辅助函数：可视化
%% =========================

function visualize_problem_comparison(problem, history_mma, fmincon_results, time_mma)
    % 可视化单个问题的结果
    
    figure('Position', [100 50 1600 900], ...
        'Name', sprintf('%s - MMA与fmincon对比', problem.name));
    
    % 1. 设计空间与MMA路径
    subplot(2, 4, [1 2]);
    plot_design_space_comparison(problem, history_mma, fmincon_results);
    title('设计空间与迭代路径', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('x_1'); ylabel('x_2'); grid on;
    
    % 2. MMA目标函数收敛
    subplot(2, 4, 3);
    plot(0:length(history_mma.f)-1, history_mma.f, 'b-o', 'LineWidth', 1.5);
    xlabel('迭代次数'); ylabel('目标函数值'); 
    title('MMA目标函数收敛', 'FontSize', 12, 'FontWeight', 'bold'); grid on;
    
    % 3. MMA可行性收敛 - 修复版本
    subplot(2, 4, 4);
    feas_history = max([zeros(1, size(history_mma.c, 2)); history_mma.c], [], 1);
    
    % 修复ylim错误：确保上限大于下限
    if all(feas_history == 0)
        % 如果所有约束违反都为0，使用固定范围
        semilogy(0:length(feas_history)-1, max(feas_history, 1e-10), 'r-o', 'LineWidth', 1.5);
        ylim([1e-11, 1e-9]);
    else
        % 否则使用自动缩放
        semilogy(0:length(feas_history)-1, max(feas_history, 1e-10), 'r-o', 'LineWidth', 1.5);
        max_val = max(feas_history);
        if max_val > 0
            ylim([1e-10, max_val * 2]);  % 使用2倍而不是1.1倍，确保有足够的空间
        end
    end
    
    xlabel('迭代次数'); ylabel('最大约束违反'); 
    title('MMA可行性收敛', 'FontSize', 12, 'FontWeight', 'bold'); grid on;
    
    % 4. 渐近线变化
    subplot(2, 4, 7);
    plot_asymptotes_comparison(history_mma);
    xlabel('迭代次数'); ylabel('值'); 
    title('渐近线变化', 'FontSize', 12, 'FontWeight', 'bold'); grid on;
    
    % 5. 子问题目标函数变化
    subplot(2, 4, 5);
    if isfield(history_mma, 'subproblem_obj') && ~isempty(history_mma.subproblem_obj)
        plot(0:length(history_mma.subproblem_obj)-1, history_mma.subproblem_obj, ...
            'm-o', 'LineWidth', 1.5);
        xlabel('迭代次数'); ylabel('子问题目标函数值'); 
        title('子问题目标函数变化', 'FontSize', 12, 'FontWeight', 'bold'); grid on;
    else
        % 如果子问题目标函数不存在，显示惩罚因子变化
        if isfield(history_mma, 'rho_history') && ~isempty(history_mma.rho_history)
            semilogy(0:length(history_mma.rho_history)-1, history_mma.rho_history, ...
                'g-o', 'LineWidth', 1.5);
            xlabel('迭代次数'); ylabel('惩罚因子ρ'); 
            title('惩罚因子变化', 'FontSize', 12, 'FontWeight', 'bold'); grid on;
        else
            % 如果都没有，显示空白
            text(0.5, 0.5, '无子问题数据', 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'FontSize', 12);
            axis off;
        end
    end
    
    % 6. 子问题约束函数变化
    subplot(2, 4, 6);
    if isfield(history_mma, 'subproblem_feas') && ~isempty(history_mma.subproblem_feas)
        plot(0:length(history_mma.subproblem_feas)-1, history_mma.subproblem_feas, ...
            'k-o', 'LineWidth', 1.5);
        xlabel('迭代次数'); ylabel('子问题最大约束违反'); 
        title('子问题约束函数变化', 'FontSize', 12, 'FontWeight', 'bold'); grid on;
    else
        % 如果子问题约束函数不存在，显示变量变化
        plot_variable_changes(history_mma);
        title('设计变量变化', 'FontSize', 12, 'FontWeight', 'bold'); grid on;
    end
    
    % 7. 算法比较
    subplot(2, 4, 8);
    plot_algorithm_comparison_bar(time_mma, fmincon_results);
    title('算法性能比较', 'FontSize', 12, 'FontWeight', 'bold'); grid on;
    
    % 添加总标题
    sgtitle(sprintf('%s - MMA算法分析', problem.name), ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', [0, 0.3, 0.6]);
end

function plot_design_space_comparison(problem, history, fmincon_results)
    % 绘制设计空间对比图
    
    x1 = linspace(problem.lb(1), problem.ub(1), 100);
    x2 = linspace(problem.lb(2), problem.ub(2), 100);
    [X1, X2] = meshgrid(x1, x2);
    
    % 计算目标函数值
    Z = zeros(size(X1));
    for i = 1:size(X1, 1)
        for j = 1:size(X1, 2)
            [Z(i, j), ~] = problem.fun([X1(i, j); X2(i, j)]);
        end
    end
    
    % 绘制等高线
    contourf(X1, X2, Z, 20, 'LineStyle', 'none');
    hold on;
    colorbar;
    xlabel('x_1'); ylabel('x_2');
    
    % 绘制MMA迭代路径
    plot(history.x(1, :), history.x(2, :), 'r-o', ...
        'LineWidth', 2, 'MarkerFaceColor', 'r', 'DisplayName', 'MMA迭代路径');
    
    % 标记起点和终点
    plot(history.x(1, 1), history.x(2, 1), 'gs', ...
        'MarkerSize', 12, 'MarkerFaceColor', 'g', 'DisplayName', '起点');
    plot(history.x(1, end), history.x(2, end), 'b*', ...
        'MarkerSize', 15, 'DisplayName', 'MMA最优解');
    
    % 标记fmincon最优点
    color_map = lines(length(fmincon_results));
    for i = 1:length(fmincon_results)
        plot(fmincon_results(i).x_opt(1), fmincon_results(i).x_opt(2), '^', ...
            'MarkerSize', 10, 'MarkerFaceColor', color_map(i, :), ...
            'DisplayName', sprintf('fmincon-%s', fmincon_results(i).algorithm));
    end
    
    % 绘制约束边界（对于非凸问题）
    if strfind(problem.name, '非凸')
        plot_constraint_boundaries(problem, X1, X2);
    end
    
    hold off;
    legend('Location', 'best');
end

function plot_constraint_boundaries(problem, X1, X2)
    % 绘制约束边界
    
    % 计算约束值
    C1 = zeros(size(X1));
    C2 = zeros(size(X1));
    
    for i = 1:size(X1, 1)
        for j = 1:size(X1, 2)
            [~, c] = problem.fun([X1(i, j); X2(i, j)]);
            C1(i, j) = c(1);
            C2(i, j) = c(2);
        end
    end
    
    % 绘制约束边界
    contour(X1, X2, C1, [0, 0], 'r-', 'LineWidth', 2, 'DisplayName', '约束1边界');
    contour(X1, X2, C2, [0, 0], 'b--', 'LineWidth', 2, 'DisplayName', '约束2边界');
    
    % 标记可行域
    feasible = (C1 <= 0) & (C2 <= 0);
    if any(feasible(:))
        contour(X1, X2, double(feasible), [0.5, 0.5], 'g-', 'LineWidth', 1, ...
            'DisplayName', '可行域');
    end
end

function plot_asymptotes_comparison(history)
    % 绘制渐近线变化
    
    L1 = history.L_asym(1, :);
    U1 = history.U_asym(1, :);
    x1_hist = history.x(1, :);
    
    plot(0:length(L1)-1, L1, 'b-', ...
         0:length(x1_hist)-1, x1_hist, 'r-', ...
         0:length(U1)-1, U1, 'b--', 'LineWidth', 1.5);
    hold on;
    
    % 填充渐近线区域
    fill_x = [0:length(L1)-1, fliplr(0:length(L1)-1)];
    fill_y = [L1, fliplr(U1)];
    fill(fill_x, fill_y, 'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    
    hold off;
    legend('L_1', 'x_1', 'U_1', 'Location', 'best');
end

function plot_algorithm_comparison_bar(time_mma, fmincon_results)
    % 绘制算法比较条形图
    
    algos = [{'MMA'}, {fmincon_results.algorithm}];
    iters = [length(fmincon_results(1).x_opt)-1, [fmincon_results.iterations]];
    times = [time_mma, [fmincon_results.time]];
    
    yyaxis left;
    bar(1:length(algos), iters, 'FaceColor', [0.2, 0.6, 0.8]);
    ylabel('迭代次数');
    
    yyaxis right;
    plot(1:length(algos), times, 'r-o', 'LineWidth', 2, 'MarkerSize', 8);
    ylabel('计算时间(s)');
    
    set(gca, 'XTick', 1:length(algos), 'XTickLabel', algos, ...
        'XTickLabelRotation', 45);
    legend('迭代次数', '计算时间', 'Location', 'best');
end

%% =========================
% 辅助函数：综合性能比较
%% =========================

function create_overall_performance_comparison(all_results)
    % 创建综合性能比较图
    
    figure('Position', [100, 100, 1200, 800], ...
        'Name', '综合性能比较', 'Color', 'white');
    
    n_problems = length(all_results);
    n_algorithms = 4;  % MMA + 3个fmincon算法
    
    % 1. 迭代次数比较
    subplot(2, 2, 1);
    plot_iteration_comparison_overall(all_results);
    title('迭代次数比较', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('迭代次数'); grid on;
    
    % 2. 计算时间比较
    subplot(2, 2, 2);
    plot_time_comparison_overall(all_results);
    title('计算时间比较', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('时间(s)'); grid on;
    
    % 3. 最优值比较
    subplot(2, 2, 3);
    plot_optimal_value_comparison_overall(all_results);
    title('最优值比较', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('目标函数值'); grid on;
    
    % 4. 可行性比较
    subplot(2, 2, 4);
    plot_feasibility_comparison_overall(all_results);
    title('可行性比较', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('最大约束违反'); grid on;
    
    sgtitle('MMA与fmincon算法综合性能比较', ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', [0, 0.3, 0.6]);
end

function plot_iteration_comparison_overall(all_results)
    % 绘制迭代次数比较
    
    n_problems = length(all_results);
    n_algorithms = 4;
    
    data = zeros(n_problems, n_algorithms);
    
    for p = 1:n_problems
        data(p, 1) = all_results(p).mma.iterations;
        for a = 1:3
            if length(all_results(p).fmincon) >= a
                data(p, a+1) = all_results(p).fmincon(a).iterations;
            end
        end
    end
    
    bar(data);
    
    % 设置x轴标签
    prob_names = cell(1, n_problems);
    for p = 1:n_problems
        prob_names{p} = sprintf('P%d\n%s', p, all_results(p).problem_name);
    end
    set(gca, 'XTickLabel', prob_names, 'FontSize', 9);
    
    legend({'MMA', 'interior-point', 'sqp', 'active-set'}, ...
        'Location', 'best', 'FontSize', 9);
end

function plot_time_comparison_overall(all_results)
    % 绘制计算时间比较
    
    n_problems = length(all_results);
    n_algorithms = 4;
    
    data = zeros(n_problems, n_algorithms);
    
    for p = 1:n_problems
        data(p, 1) = all_results(p).mma.time;
        for a = 1:3
            if length(all_results(p).fmincon) >= a
                data(p, a+1) = all_results(p).fmincon(a).time;
            end
        end
    end
    
    bar(data);
    
    prob_names = cell(1, n_problems);
    for p = 1:n_problems
        prob_names{p} = sprintf('P%d', p);
    end
    set(gca, 'XTickLabel', prob_names, 'FontSize', 9);
end

function plot_optimal_value_comparison_overall(all_results)
    % 绘制最优值比较
    
    n_problems = length(all_results);
    n_algorithms = 4;
    
    data = zeros(n_problems, n_algorithms);
    
    for p = 1:n_problems
        data(p, 1) = all_results(p).mma.f_opt;
        for a = 1:3
            if length(all_results(p).fmincon) >= a
                data(p, a+1) = all_results(p).fmincon(a).f_opt;
            end
        end
    end
    
    bar(data);
    
    prob_names = cell(1, n_problems);
    for p = 1:n_problems
        prob_names{p} = sprintf('P%d', p);
    end
    set(gca, 'XTickLabel', prob_names, 'FontSize', 9);
    
    % 添加标签
    for p = 1:n_problems
        for a = 1:n_algorithms
            if data(p, a) > 0
                text(p-0.3+(a-1)*0.2, data(p, a)+max(data(:))*0.01, ...
                    sprintf('%.3f', data(p, a)), 'FontSize', 8, ...
                    'HorizontalAlignment', 'center');
            end
        end
    end
end

function plot_feasibility_comparison_overall(all_results)
    % 绘制可行性比较
    
    n_problems = length(all_results);
    n_algorithms = 4;
    
    data = zeros(n_problems, n_algorithms);
    
    for p = 1:n_problems
        data(p, 1) = all_results(p).mma.feasibility;
        for a = 1:3
            if length(all_results(p).fmincon) >= a
                data(p, a+1) = all_results(p).fmincon(a).feasibility;
            end
        end
    end
    
    % 使用对数坐标
    semilogy(1:n_problems, data, 'o-', 'LineWidth', 1.5, 'MarkerSize', 8);
    
    prob_names = cell(1, n_problems);
    for p = 1:n_problems
        prob_names{p} = sprintf('P%d', p);
    end
    set(gca, 'XTick', 1:n_problems, 'XTickLabel', prob_names, 'FontSize', 9);
    
    legend({'MMA', 'interior-point', 'sqp', 'active-set'}, ...
        'Location', 'best', 'FontSize', 9);
    
    % 添加可行性阈值线
    hold on;
    plot([0, n_problems+1], [1e-6, 1e-6], 'k--', 'LineWidth', 1);
    text(n_problems/2, 2e-6, '可行性阈值 (1e-6)', 'FontSize', 9);
    hold off;
end

%% =========================
% 辅助函数：包装函数
%% =========================

function f = fmincon_obj_wrapper(x, fun)
    [f, ~, ~, ~] = fun(x);
end

function [c, ceq] = fmincon_con_wrapper(x, fun)
    [~, c, ~, ~] = fun(x);
    ceq = [];
end
function plot_mma_subproblem_evolution(problem, history)

% 选择几个关键迭代
K = length(history.subprob);
if K < 3
    idx = 1:K;
else
    idx = unique(round([1, K/2, K]));
end

x1 = linspace(problem.lb(1), problem.ub(1), 200);
x2 = linspace(problem.lb(2), problem.ub(2), 200);
[X1, X2] = meshgrid(x1, x2);

hold on;

colors = lines(length(idx));

for t = 1:length(idx)
    k = idx(t);
    model = history.subprob{k};

    F = zeros(size(X1));
    G = zeros(size(X1));

    for i = 1:numel(X1)
        y = [X1(i); X2(i)];
        F(i) = mma_eval_obj(y, model);
        G(i) = mma_eval_con(y, model);
    end

    % 画目标等高线
    contour(X1, X2, F, 15, 'LineColor', colors(t,:), ...
        'DisplayName', sprintf('Subproblem obj @ iter %d', k));

    % 画可行边界
    contour(X1, X2, G, [0 0], '--', 'LineWidth', 2, ...
        'LineColor', colors(t,:), ...
        'DisplayName', sprintf('Subproblem feas @ iter %d', k));

    % 标记展开点
    plot(model.xk(1), model.xk(2), 'o', ...
        'Color', colors(t,:), ...
        'MarkerFaceColor', colors(t,:), ...
        'DisplayName', sprintf('Expand pt @ iter %d', k));
end

% 画真实MMA路径
plot(history.x(1,:), history.x(2,:), 'k-o', ...
    'LineWidth', 2, 'DisplayName', 'MMA trajectory');

xlabel('x1'); ylabel('x2');
title('MMA Subproblem Evolution');
grid on; axis equal;
legend('Location','bestoutside');
end
function f = mma_eval_obj(y, model)
f = model.r0;
for j = 1:length(y)
    f = f + model.p0(j)/(model.U(j)-y(j)) ...
          + model.q0(j)/(y(j)-model.L(j));
end
end


function g = mma_eval_con(y, model)
m = length(model.R);
gvals = zeros(m,1);
for i = 1:m
    gvals(i) = model.R(i);
    for j = 1:length(y)
        gvals(i) = gvals(i) ...
            + model.P(j,i)/(model.U(j)-y(j)) ...
            + model.Q(j,i)/(y(j)-model.L(j));
    end
end
g = max(gvals);
end

