function direction = Hua_direction(M, W)
samples = construct_polar(M-1, W);
direction = ones(W, M);
S = 2*pi^(M/2)/gamma(M/2)/(2^M);
int_all = zeros(1, M-1);
for i=1:M-1
    int_all(i) = int_of_power_of_sin(M-i-1, 0, pi/2);
end
prod_int_all = prod(int_all);
ints = zeros(1, M-1);
for i=1:M-1
    ints(i) = prod_int_all/int_all(i);
end
ints = ints/S;
for i=1:W
    for j=1:M-1
            theta(j) = solve_inverse_int_of_power_sin(samples(i, j)/ints(j), M-j-1);
    end
    for j=1:M       % j-th component of i-th direction
        for k=1:M-j
            direction(i, j) = direction(i, j)*sin(theta(k));
        end
        if j>1
            direction(i, j) = direction(i, j)*cos(theta(M-j+1));
        end
    end
end

function x = solve_inverse_int_of_power_sin(theta, dim)
x = pi/2;
newf = int_of_power_of_sin(dim, 0, x) - theta;
while abs(newf)>1e-10
    g = sin(x)^dim;
    x = x-newf/g;
    newf = int_of_power_of_sin(dim, 0, x) - theta;
end

function grids = construct_polar(s, n)
% construct n points in [0, 1]^s (Hua-Wang method)
% Step 1: find prime p such that s<=eularfunction(p)/2=(p-1)/2
p = s*2+1; flag = 0;
while flag==0
    flag = 1;
    for i=3:floor(sqrt(p))
        if p/i==floor(p/i)
            flag = 0;
            break;
        end
    end
    if flag==0
        p = p+2;
    end
end

% Step 2: construct a(n) according to original s
a(1) = 1;
for i=2:s
    temp = 2*cos(2*pi*(i-1)/p);
    a(i) = round(n*(abs(temp) - floor(abs(temp))));
end

% Step 3: construct samping points
grids = zeros(n, s);
for i=1:n
    grids(i, :) = i*a/n;
    grids(i, :) = grids(i, :) - floor(grids(i, :));
end