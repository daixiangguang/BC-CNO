function [index,obj]=BKmeans(X,k)
[n,m]=size(X); %m表示样本个数,n表示特征个数
rd=randperm(n);
C=X(rd(1:k),:);
is=repmat(1:k,1,n/k);
iter=20;
obj=zeros(iter,1);
for i=1:iter
% Assignment step
[p,obj(i)]=assignment(X,C,n,k,is);

%Update step
C=update(X,p,n,k,is);
end
index=zeros(n,1);
for i=1:n
    index(i)=find(p(i,:)==1);
end

for i=1:length(is)
    index(i)=is(index(i));
end



function [p,obj]=assignment(X,C,n,k,is)
% [m2,n2]=size(C);
W=zeros(n,n); %m1表示样本个数，m2表示类个数
parfor a=1:n
    for i=1:n
        W(a,i)=norm(X(i,:)-C(is(a),:),2)^2;
    end
end
% [p,obj]=Hungarian (W');
[p,obj] = munkres(W');
function C=update(X,p,n,k,is)
index=zeros(n,1);
for i=1:n
    index(i)=find(p(i,:)==1);
end

for i=1:length(is)
    index(i)=is(index(i));
end
% index=mod(index,k)+1;
C=zeros(k,size(X,2));
for i=1:k
    C(i,:)=mean(X(index==i,:),1);
end