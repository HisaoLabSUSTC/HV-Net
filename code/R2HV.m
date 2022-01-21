function R2val = R2HV(data,ref,lineNum)
    [N,dim] = size(data);
    [V,row] = UniformVector(lineNum,dim);
    %[row,dim] = size(V);
    y = 0;
    for j=1:row
        temp = abs(data-ref)./V(j,:);
        [x,~] = max(min(temp,[],2));
        y = y+x^dim;
    end
    R2val = y*pi^(dim/2)/(dim*row*2^(dim-1)*gamma(dim/2));
end