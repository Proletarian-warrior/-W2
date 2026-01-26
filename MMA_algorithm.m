function [x_opt, f_opt, history] = MMA_algorithm(fun, x0, lb, ub, options)
% =========================================================================
% Enhanced MMA (Method of Moving Asymptotes) - Improved convergence
% + Records full subproblem models (for visualization / analysis)
% =========================================================================

if nargin < 5, options = struct(); end
if ~isfield(options,'max_iter'), options.max_iter = 50; end
if ~isfield(options,'tol'), options.tol = 1e-6; end
if ~isfield(options,'s0'), options.s0 = 0.3; end
if ~isfield(options,'display'), options.display = true; end
if ~isfield(options,'rho0'), options.rho0 = 1.0; end
if ~isfield(options,'rho_max'), options.rho_max = 1e6; end

x = x0(:);
n = length(x);
L = lb(:);
U = ub(:);

% ===== 初始渐近线 =====
L_asym = x - options.s0 * (U - L);
U_asym = x + options.s0 * (U - L);

% ===== 历史记录 =====
history.x = x;
history.f = [];
history.c = [];
history.L_asym = L_asym;
history.U_asym = U_asym;
history.subproblem_obj = [];
history.subproblem_feas = [];
history.rho_history = [];
history.subprob = {};    % ★ 用 cell 存储每一步子问题模型（关键修复）

% ===== 初始评估 =====
[f, c, df, dc] = fun(x);
m = length(c);

rho = options.rho0;
feas_old = max([0; c]);

if options.display
    fprintf('\n==== Enhanced MMA Started ====\n');
    fprintf('%4s %14s %14s %10s %10s\n','Iter','f','max(c)','rho','step');
    fprintf('%4d %14.6e %14.6e %10.2e %10.2e\n',0,f,feas_old,rho,0);
end

x_old = x;
x_older = x;

no_improvement_count = 0;
best_f = f;
best_feas = feas_old;

for k = 1:options.max_iter
    % ===== 惩罚函数 =====
    active_constraints = c > 0;
    num_active = sum(active_constraints);
    
    f_pen = f + rho * sum(max(c, 0));
    
    % ===== 梯度计算 =====
    if num_active > 0
        df_pen = df;
        for i = find(active_constraints)'
            df_pen = df_pen + rho * dc(:, i);
        end
    else
        df_pen = df;
    end
    
    % ===== 构造MMA子问题 =====
    [sub_fun, sub_con, alpha, beta, submodel] = build_subproblem_enhanced( ...
        x, f_pen, c, df_pen, dc, L_asym, U_asym, L, U);
    
    % ===== 保存子问题模型（关键功能）=====
    history.subprob{end+1} = submodel;
    
    % ===== 解子问题 =====
    options_fmincon = optimoptions('fmincon', ...
        'Algorithm', 'sqp-legacy', ...
        'Display', 'none', ...
        'OptimalityTolerance', 1e-8, ...
        'StepTolerance', 1e-8, ...
        'MaxIterations', 100);
    
    try
        [x_new, fsub, exitflag] = fmincon(sub_fun, x, [], [], [], [], ...
            alpha, beta, sub_con, options_fmincon);
        
        if exitflag <= 0
            x_new = 0.5 * (x + 0.5 * (alpha + beta));
        end
    catch
        x_new = 0.5 * (x + 0.5 * (alpha + beta));
    end
    
    % ===== 新点评估 =====
    [f_new, c_new, df_new, dc_new] = fun(x_new);
    feas_new = max([0; c_new]);
    
    % ===== 收敛判断 =====
    dx_norm = norm(x_new - x) / max(1, norm(x));
    df_rel = abs(f_new - f) / max(1, abs(f));
    
    is_converged = (dx_norm < options.tol && ...
                   df_rel < options.tol && ...
                   feas_new < options.tol);
    
    % 监控改进
    if feas_new <= best_feas && f_new < best_f
        best_f = f_new;
        best_feas = feas_new;
        no_improvement_count = 0;
    else
        no_improvement_count = no_improvement_count + 1;
    end
    
    if no_improvement_count > 5
        if feas_new > 1e-3 && rho < options.rho_max
            rho = min(rho * 2, options.rho_max);
        end
    end
    
    if options.display && mod(k, 5) == 0
        fprintf('%4d %14.6e %14.6e %10.2e %10.2e\n', ...
            k, f_new, feas_new, rho, dx_norm);
    end
    
    if is_converged
        x = x_new; f = f_new; c = c_new; df = df_new; dc = dc_new;
        if options.display
            fprintf('%4d %14.6e %14.6e %10.2e %10.2e  (CONVERGED)\n', ...
                k, f, feas_new, rho, dx_norm);
        end
        break;
    end
    
    if rho > 1e6 && feas_new > 1e-3 && no_improvement_count > 10
        if options.display
            fprintf('%4d %14.6e %14.6e %10.2e %10.2e  (EARLY STOP)\n', ...
                k, f_new, feas_new, rho, dx_norm);
        end
        break;
    end
    
    % ===== 更新渐近线 =====
    if k >= 2
        [L_asym, U_asym] = update_asymptotes_enhanced(x_new, x, x_old, ...
            L_asym, U_asym, L, U, k);
    else
        L_asym = x_new - options.s0 * (U - L);
        U_asym = x_new + options.s0 * (U - L);
    end
    
    % ===== 更新变量 =====
    x_older = x_old;
    x_old = x;
    x = x_new;
    
    f = f_new;
    c = c_new;
    df = df_new;
    dc = dc_new;
    feas_old = feas_new;
    
    % ===== 保存历史 =====
    history.x(:, end+1) = x;
    history.f(end+1) = f;
    history.c(:, end+1) = c;
    history.L_asym(:, end+1) = L_asym;
    history.U_asym(:, end+1) = U_asym;
    history.subproblem_obj(end+1) = fsub;
    history.subproblem_feas(end+1) = feas_new;
    history.rho_history(end+1) = rho;
end

x_opt = x;
f_opt = f;

final_feas = max([0; c]);
if options.display
    fprintf('\n==== Enhanced MMA Finished ====\n');
    fprintf('Final: f = %.6f, max(c) = %.2e, iterations = %d\n', ...
        f_opt, final_feas, size(history.x, 2)-1);
    
    if final_feas > 1e-3
        fprintf('Warning: Constraints not fully satisfied (max violation = %.2e)\n', final_feas);
    end
end

end

%% ========================================================
% 改进的子问题构建（带模型输出）
%% ========================================================
function [sub_fun, sub_con, alpha, beta, submodel] = build_subproblem_enhanced( ...
    x, f, c, df, dc, Lasy, Uasy, L, U)

n = length(x);
m = length(c);
eps_safe = 1e-8;

Lasy = max(Lasy, L + 1e-6);
Uasy = min(Uasy, U - 1e-6);

% ===== 计算p,q系数 =====
p0 = zeros(n, 1);
q0 = zeros(n, 1);
for j = 1:n
    if df(j) > 0
        p0(j) = df(j) * (Uasy(j) - x(j))^2;
    else
        q0(j) = -df(j) * (x(j) - Lasy(j))^2;
    end
end

r0 = f;
for j = 1:n
    if p0(j) > 0
        r0 = r0 - p0(j) / (Uasy(j) - x(j));
    end
    if q0(j) > 0
        r0 = r0 - q0(j) / (x(j) - Lasy(j));
    end
end

p_con = zeros(n, m);
q_con = zeros(n, m);
r_con = zeros(m, 1);

for i = 1:m
    for j = 1:n
        if dc(j, i) > 0
            p_con(j, i) = dc(j, i) * (Uasy(j) - x(j))^2;
        else
            q_con(j, i) = -dc(j, i) * (x(j) - Lasy(j))^2;
        end
    end
    
    r_con(i) = c(i);
    for j = 1:n
        if p_con(j, i) > 0
            r_con(i) = r_con(i) - p_con(j, i) / (Uasy(j) - x(j));
        end
        if q_con(j, i) > 0
            r_con(i) = r_con(i) - q_con(j, i) / (x(j) - Lasy(j));
        end
    end
end

sub_fun = @(y) mma_frac_enhanced(y, p0, q0, r0, Lasy, Uasy, eps_safe);
sub_con = @(y) deal(mma_frac_con_enhanced(y, p_con, q_con, r_con, Lasy, Uasy, eps_safe), []);

move_factor = 0.3;
alpha = max([L, x - move_factor * (U - L), Lasy + 0.1 * (x - Lasy)], [], 2);
beta  = min([U, x + move_factor * (U - L), Uasy - 0.1 * (Uasy - x)], [], 2);

alpha = max(alpha, L);
beta  = min(beta, U);


submodel = struct();
submodel.sub_fun = sub_fun;
submodel.sub_con = sub_con;
submodel.alpha = alpha;
submodel.beta = beta;

% ===== MMA解析模型参数 =====
submodel.p0 = p0;
submodel.q0 = q0;
submodel.r0 = r0;

submodel.P = p_con;     % 统一接口名
submodel.Q = q_con;
submodel.R = r_con;

submodel.L = Lasy;      % 统一接口名
submodel.U = Uasy;

submodel.xk = x;        % 当前展开点

end

function val = mma_frac_enhanced(y, p, q, r, L, U, eps_safe)
n = length(y);
val = r;
for j = 1:n
    den1 = max(U(j) - y(j), eps_safe);
    den2 = max(y(j) - L(j), eps_safe);
    if p(j) > 0, val = val + p(j) / den1; end
    if q(j) > 0, val = val + q(j) / den2; end
end
end

function c_vals = mma_frac_con_enhanced(y, p, q, r, L, U, eps_safe)
m = size(p, 2);
n = length(y);
c_vals = zeros(m, 1);
for i = 1:m
    c_vals(i) = r(i);
    for j = 1:n
        den1 = max(U(j) - y(j), eps_safe);
        den2 = max(y(j) - L(j), eps_safe);
        if p(j, i) > 0, c_vals(i) = c_vals(i) + p(j, i) / den1; end
        if q(j, i) > 0, c_vals(i) = c_vals(i) + q(j, i) / den2; end
    end
end
end

%% ========================================================
% 增强的渐近线更新
%% ========================================================
function [Lnew, Unew] = update_asymptotes_enhanced(xnew, x, xold, Lold, Uold, L, U, iter)

n = length(xnew);
Lnew = zeros(n, 1);
Unew = zeros(n, 1);

for j = 1:n
    d1 = xnew(j) - x(j);
    d2 = x(j) - xold(j);
    
    if abs(d1) < 1e-8 || abs(d2) < 1e-8
        factor = 0.8;
    elseif d1 * d2 > 0
        factor = 1.1;
    else
        factor = 0.6;
    end
    
    Lnew(j) = max(xnew(j) - factor * (x(j) - Lold(j)), L(j) + 1e-6);
    Unew(j) = min(xnew(j) + factor * (Uold(j) - x(j)), U(j) - 1e-6);
    
    if Unew(j) - Lnew(j) < 1e-6
        Lnew(j) = L(j);
        Unew(j) = U(j);
    end
end
end
