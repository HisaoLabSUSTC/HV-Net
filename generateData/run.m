% 这个代码是用来产生训练数据和测试数据

% -----------task-------------------
% dataset_num 100,000 -> 1,000,000
% dataset:
%   正三角PF上: 50,000
%   倒三角PF上: 50,000
%   random generated: 900,000
% -----------task-------------------

% -----------train-------------------
target          = 'train';
data_num        = 100; %每个solution set包含的解的数量
dataset_num     = 1000000; %一共有这么多solution set
num_on_triPF    = 0;     % 这么多solution set从triangular PF上生成
num_on_invtriPF = 0;     % 这么多solution set从inverted triangular PF上生成
num_on_random   = 1000000; % 这么多solution set random 生成后选取适当多的non-dominated point
% -----------train-------------------

% % -----------train-------------------
% target          = 'train';
% data_num        = 100; %每个solution set包含的解的数量
% dataset_num     = 200000; %一共有这么多solution set
% num_on_triPF    = 50000;     % 这么多solution set从triangular PF上生成
% num_on_invtriPF = 50000;     % 这么多solution set从inverted triangular PF上生成
% num_on_random   = 100000; % 这么多solution set random 生成后选取适当多的non-dominated point
% % -----------train-------------------

% % -----------test-------------------
% target          = 'train';
% data_num        = 100; %每个solution set包含的解的数量
% dataset_num     = 10000; %一共有这么多solution set
% num_on_triPF    = 0;     % 这么多solution set从triangular PF上生成
% num_on_invtriPF = 0;     % 这么多solution set从inverted triangular PF上生成
% num_on_random   = 10000; % 这么多solution set random 生成后选取适当多的non-dominated point
% % -----------test-------------------

M = 5; %目标个数
seeds = 3:5;   % 

for seed=seeds
    r = 1;
    Data = ones(dataset_num,data_num,M)*nan;
    HVval = zeros(dataset_num,1);
    % generate solution set and HVC
    Data = generateTrainingData(M,Data,data_num,num_on_triPF,num_on_invtriPF,num_on_random,seed);
    %Data(1,1,1)
    rng('shuffle');
    for i=1:dataset_num
        if mod(i, 10000) == 0
            disp(['HVcal, i=',num2str(i),'/',num2str(dataset_num)]);
            toc
        end
        data = Data(i,:,:);
        data = data(1,~isnan(data(1,:,1)),:);
        %b=Data(~isnan(a(:,1)),:);
        HVval(i,1) = HV(data,r);  
    end
    %保存数据，Data是solution sets，HVval是对应的hypervolume值
    save(['//10.20.2.245/datasets/HV-Net-datasets/', target, '_data_M', num2str(M), '_', num2str(seed), '.mat'],'Data','HVval');
end