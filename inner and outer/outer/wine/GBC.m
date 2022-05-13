function l=GBC(X,c,lambda,gamma,mu,iter)
%X R^{d n}
%b R^{c 1}
%Y {0,1}^{n c}
%W R^{d c}
X=X';
obj=zeros(iter,1);
[d,n]=size(X);
In=eye(n,n);
Id=eye(d,d);
Y=randn(n,c);
Y(Y>=0)=1;
Y(Y<0)=0;
W=rand(d,c);
I=ones(n,1);
eta=rand(n,c);
ro=1.005;
aa=(1:n)';
for i=1:iter
    % Çób
     b=1/n*(Y'*I-W'*X*I);
    
    %ÇóW
    Hc=In-1/n*I*I';
%     A=inv(X*Hc*X'+gamma*Id);
    Xc=X*Hc;
%     W=A*Xc*Y;
    
    % ÇóY
    L0=Hc-Xc'*inv((Xc*Xc'+gamma*Id))*Xc+lambda*I*I';
    M=L0;
    for k=1:50
        Z=inv(mu*In+2*M)*(mu*Y+eta);
        V=Z-1/mu*eta;
        [~,cc]=max(V');
        cc=cc';
        index=[aa cc];
        newindex=(index(:,2)-1)*n+index(:,1);
        Y=zeros(n,c);
        Y(newindex)=1;
        eta=eta+mu*(Y-Z);
        mu=ro*mu;
    end
%     obj(i)=norm(X'*W+I*b'-Y,'fro')^2+gamma*norm(W,'fro')^2+lambda*trace(Y'*I*I'*Y);
    
end
l=label(Y,n);
function l=label(x,n)
l=zeros(n,1);
for i=1:n
    l(i)=find(x(i,:)==1);
end