function d=gaussinKernel(X,sigma)
[n,m]=size(X);%n样本个数，m特征维度
d=ones(n,n);
   X=X-sum(X,2)/n;
for i=1:n-1
    for j=i+1:n
        x=X(i,:)-X(j,:);
        d(i,j)=exp(-x*x'/(2*sigma*sigma));
        d(j,i)=d(i,j);
    end
end
% d=d-sum(d,2)/n;