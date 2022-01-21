% test_file = 'test_data_tri_2';
% test_file = 'test_data_invtri_2';
test_file = 'test_data_random_2';
load(['//10.20.2.245/datasets/HV-Net-datasets/', test_file, '.mat']);

[L,N,M]  = size(Data);
sampleNum = 100:100:2000;
%sampleNum = 2000;
lineNum = 10:10:200;
runs = 10;

Time = zeros(1,length(sampleNum));
Loss = zeros(1,length(sampleNum));
Time1 = zeros(1,length(lineNum));
Loss1 = zeros(1,length(lineNum));
% 
for k=1:length(sampleNum)
    k
    loss = 0;
    tic;
    for i=1:L
       pop = Data(i,:,:);
       pop = reshape(pop,N,M);
       pop = pop(~isnan(pop(:,1)),:);
       hvMC = MC(pop,1,sampleNum(k));
       %loss = loss+(hvMC-HVval(i))^2;  
       loss = loss+abs(hvMC-HVval(i))/HVval(i);
    end
    loss = loss/L;
    time = toc;
    Loss(k) = loss;
    Time(k) = time;
end



for k=1:length(lineNum)
    k
    loss = 0;
    tic;
    for i=1:L
       pop = Data(i,:,:);
       pop = reshape(pop,N,M);
       pop = pop(~isnan(pop(:,1)),:);
       hvMC = R2HV(pop,1,lineNum(k));
       %loss = loss+(hvMC-HVval(i))^2;  
       loss = loss+abs(hvMC-HVval(i))/HVval(i);
    end
    loss = loss/L;
    time = toc;
    Loss1(k) = loss;
    Time1(k) = time;
end

% 
% tic
% for i=1:L
%    pop = Data(i,:,:);
%    pop = reshape(pop,N,M);
%    pop = pop(~isnan(pop(:,1)),:);
%    hvMC = R2HV(pop,1,lineNum);
%    loss = loss+abs(hvMC-HVval(i))/HVval(i);  
% end
% loss = loss/L;
% toc

save(['//10.20.2.245/datasets/HV-Net-datasets/result_MC_R2_', test_file, '.mat'], 'Loss', 'Time', 'Loss1', 'Time1');
