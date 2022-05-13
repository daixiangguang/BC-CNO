function  [w1,const1,w2,const2]=getParameters(d,n,p,u)




blocks=zeros(n,n*p);
for i=1:n
    block=zeros(1,n);
    block(i)=1;
    block=repmat(block,1,p);
    blocks(i,:)=block;
end
% 复制p个块
w1=repmat(blocks,p,1);

%确定theta2
theta1=-2*ones(n*p,1);  %这个是一次项的系数 
w1=w1+diag(theta1);
const1=n;






w2=repmat(blocks,p,1);
for i=1:p
    w2((i-1)*n+1:i*n,(i-1)*n+1:i*n)=1;
end
%ȷ��theta3
theta2=-2*ones(n*p,1)*u;
w2=w2+diag(theta2);
const2=p*u*u;