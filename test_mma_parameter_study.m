%% ========================================================
% MMA 参数敏感性分析实验 
% 分析参数对：收敛速度、精度、稳定性的影响
%% ========================================================

clear; close all; clc;

fprintf('=== MMA 参数敏感性分析实验 ===\n');

%% ===== 选用测试问题（两杆桁架）=====
problem.name = '两杆桁架';
problem.fun = @two_bar_truss_standard;
problem.x0 = [1.5; 0.5];
problem.lb = [0.1; 0.1];
problem.ub = [2; 1];

%% ===== 先用 fmincon 求一个“高精度参考解”作为真值 =====
fprintf('使用 fmincon 计算参考最优解...\n');

opts_ref = optimoptions('fmincon', ...
    'Algorithm','sqp', ...
    'Display','off', ...
    'OptimalityTolerance',1e-12, ...
    'ConstraintTolerance',1e-12);

[x_ref, f_ref] = fmincon( ...
    @(x) fmincon_obj_wrapper(x, problem.fun), ...
    problem.x0, [], [], [], [], ...
    problem.lb, problem.ub, ...
    @(x) fmincon_con_wrapper(x, problem.fun), ...
    opts_ref);

fprintf('参考解: f = %.10f, x = [%.6f, %.6f]\n', f_ref, x_ref(1), x_ref(2));

%% ===== 扫描参数 =====
s0_list   = [0.1, 0.2, 0.3, 0.4, 0.5];
rho0_list = [0.1, 1, 10];
tol_list  = [1e-3, 1e-5, 1e-7];

%% ===== 结果存储 =====
RESULT = struct();
cnt = 0;

for s0 = s0_list
    for rho0 = rho0_list
        for tol = tol_list
            cnt = cnt + 1;

            fprintf('Test %d: s0=%.2f, rho0=%.2f, tol=%.1e\n', ...
                cnt, s0, rho0, tol);

            mma_options = struct( ...
                'max_iter', 200, ...
                'tol', tol, ...
                's0', s0, ...
                'display', false, ...
                'rho0', rho0, ...
                'rho_max', 1e6);

            tic;
            [x_opt, f_opt, history] = MMA_algorithm( ...
                problem.fun, problem.x0, problem.lb, problem.ub, mma_options);
            time_cost = toc;

            [~, c, ~, ~] = problem.fun(x_opt);
            feas = max([0; c]);

            % ===== 新增：与参考解的距离（最终收敛位置精度）=====
            x_err = norm(x_opt - x_ref);

            RESULT(cnt).s0 = s0;
            RESULT(cnt).rho0 = rho0;
            RESULT(cnt).tol = tol;
            RESULT(cnt).f = f_opt;
            RESULT(cnt).iter = length(history.f)-1;
            RESULT(cnt).time = time_cost;
            RESULT(cnt).feas = feas;
            RESULT(cnt).x_err = x_err;   % ★ 新增
        end
    end
end

save('mma_param_study.mat', 'RESULT', 'x_ref', 'f_ref');

fprintf('参数实验完成，结果已保存。\n');

%% ========================================================
% 可视化分析
%% ========================================================

%% 转为表格
T = struct2table(RESULT);

%% 1️⃣ s0 对迭代次数影响
figure;
for rho0 = rho0_list
    idx = T.rho0 == rho0 & T.tol == 1e-5;
    plot(T.s0(idx), T.iter(idx), '-o', 'LineWidth', 2); hold on;
end
grid on;
xlabel('初始渐近线系数 s0');
ylabel('迭代次数');
title('参数 s0 对 MMA 收敛速度的影响');
legend("rho0=0.1","rho0=1","rho0=10", 'Location','best');

%% 2️⃣ s0 对可行性精度影响
figure;
for rho0 = rho0_list
    idx = T.rho0 == rho0 & T.tol == 1e-5;
    semilogy(T.s0(idx), T.feas(idx), '-o', 'LineWidth', 2); hold on;
end
grid on;
xlabel('初始渐近线系数 s0');
ylabel('最大约束违反');
title('参数 s0 对约束违反的影响');
legend("rho0=0.1","rho0=1","rho0=10", 'Location','best');

%% 3️⃣ tol 对迭代次数影响
figure;
for s0 = s0_list
    idx = T.s0 == s0 & T.rho0 == 1;
    semilogx(T.tol(idx), T.iter(idx), '-o', 'LineWidth', 2); hold on;
end
grid on;
xlabel('收敛容差 tol');
ylabel('迭代次数');
title('收敛容差 tol 对 MMA 收敛速度的影响');
legend("s0=0.1","s0=0.2","s0=0.3","s0=0.4","s0=0.5", 'Location','best');

%% 4️⃣ ★ 新增：s0 对“最终收敛位置精度”的影响
figure;
for rho0 = rho0_list
    idx = T.rho0 == rho0 & T.tol == 1e-5;
    semilogy(T.s0(idx), T.x_err(idx), '-o', 'LineWidth', 2); hold on;
end
grid on;
xlabel('初始渐近线系数 s0');
ylabel('||x_{MMA} - x_{ref}||_2');
title('参数 s0 对最终解精度的影响');
legend("rho0=0.1","rho0=1","rho0=10", 'Location','best');

fprintf('全部图像已生成。\n');

%% ========================================================
% fmincon 包装函数
%% ========================================================

function f = fmincon_obj_wrapper(x, fun)
    [f, ~, ~, ~] = fun(x);
end

function [c, ceq] = fmincon_con_wrapper(x, fun)
    [~, c, ~, ~] = fun(x);
    ceq = [];
end
