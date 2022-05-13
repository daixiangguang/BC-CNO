function [hat_W,theta_,w1,const1,w2,const2]=getCQPParameterst(d,n,p,u,alpha,beta)
% w1,w2,w3,theta2,theta3
% ��QAP����ת����CQP���⣬��ȡCQP�����W��theta
% n������
% p����
w1=zeros(n*p,n*p);
w2=zeros(n*p,n*p);
% theta2=zeros(n*p,1);
% theta3=zeros(n*p,1);
w3=zeros(n*p,n*p);

%ȷ��w1
for i=1:p
    w1((i-1)*n+1:i*n,(i-1)*n+1:i*n)=d;
end

%ȷ��w2
blocks=zeros(n,n*p);
for i=1:n
    block=zeros(1,n);
    block(i)=1;
    block=repmat(block,1,p);
    blocks(i,:)=block;
end
% ����p����
w2=alpha*repmat(blocks,p,1);

%ȷ��theta2
theta2=-alpha*ones(n*p,1);

%ȷ��w3
for i=1:p
    w3((i-1)*n+1:i*n,(i-1)*n+1:i*n)=1;
end
w3=beta*w3;
%ȷ��theta3
theta3=-beta*ones(n*p,1)*u;

%w1=sparse(w1);
%w2=sparse(w2);
%w3=sparse(w3);


WD11=diag(w1);wd11=diag(WD11);
WD22=diag(w2);wd22=diag(WD22);
WD33=diag(w3);wd33=diag(WD33);
w11=w1-wd11;w22=w2-wd22;w33=w3-wd33;
hat_W=w11+w22+w33; 
theta_=0.5*WD11'+0.5*WD22'+0.5*WD33'+theta2'+theta3';

 [w1,const1,w2,const2]=getParameters(d,n,p,u);
