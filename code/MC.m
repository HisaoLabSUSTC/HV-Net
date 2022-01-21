function Score = MC(PopObj,r,SampleNum)
% Estimate the HV value by Monte Carlo estimation
    [N,M]  = size(PopObj);
    RefPoint = ones(1,M)*r;
    MaxValue  = RefPoint;
    MinValue  = min(PopObj,[],1);
    Samples   = unifrnd(repmat(MinValue,SampleNum,1),repmat(MaxValue,SampleNum,1));
%         if gpuDeviceCount > 0
%             % GPU acceleration
%             Samples = gpuArray(single(Samples));
%             PopObj  = gpuArray(single(PopObj));
%         end
    for i = 1 : size(PopObj,1)
        %drawnow();
        domi = true(size(Samples,1),1);
        m    = 1;
        while m <= M && any(domi)
            domi = domi & PopObj(i,m) <= Samples(:,m);
            m    = m + 1;
        end
        Samples(domi,:) = [];
    end
    Score = prod(MaxValue-MinValue)*(1-size(Samples,1)/SampleNum);
        
end