load('//10.20.2.245/datasets/HV-Net-datasets/test_data_tri_2.mat');

[L,N,M]  = size(Data);
sampleNum = 2000;
lineNum = 200;
loss = 0;

tic
for i=1:L
   pop = Data(i,:,:);
   pop = reshape(pop,N,M);
   pop = pop(~isnan(pop(:,1)),:);
   hvMC = MC(pop,1,sampleNum);
   %loss = loss+(hvMC-HVval(i))^2;  
   loss = loss+abs(hvMC-HVval(i))/HVval(i);
end
loss = loss/L;
toc

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