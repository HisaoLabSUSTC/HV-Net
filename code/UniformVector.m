function [V,N] = UniformVector(N,M,seed)
% Generate uniformly distributed direction vectors

%--------------------------------------------------------------------------
% Author: Ke Shang
% The code uses PlatEMO published in "Ye Tian, Ran Cheng, Xingyi Zhang, 
% and Yaochu Jin, PlatEMO: A MATLAB Platform for Evolutionary 
% Multi-Objective Optimization [Educational Forum], IEEE Computational 
% Intelligence Magazine, 2017, 12(4): 73-87".
%-------------------------------------------------------------------------- 
    %rng(seed);
    mu = zeros(1,M);
    sigma = eye(M,M);
    R = mvnrnd(mu,sigma,N);
    V = abs(R./sqrt(sum(R.^2,2)));
end