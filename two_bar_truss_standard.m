function [f, c, df, dc] = two_bar_truss_standard(x)

x1 = x(1);
x2 = x(2);

f = x1 * sqrt(1 + x2^2);

c = 0.124 * sqrt(1 + x2^2) * (8/x1 + 1/(x1*x2)) - 1;

df = [ sqrt(1 + x2^2);
       x1 * x2 / sqrt(1 + x2^2) ];

dc = zeros(2,1);
dc(1) = 0.124 * sqrt(1 + x2^2) * (-8/x1^2 - 1/(x1^2*x2));
dc(2) = 0.124 * ( ...
          (x2/sqrt(1 + x2^2))*(8/x1 + 1/(x1*x2)) ...
          - sqrt(1 + x2^2)/(x1*x2^2) );
end