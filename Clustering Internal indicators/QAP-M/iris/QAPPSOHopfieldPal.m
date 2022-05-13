function [l,y]=QAPPSOHopfieldPal(X,options)
%the number of cpu cores:cores
%the number of population in each cpu core; pop
%the population size: sizepop=cores*pop
%the iteration counts: maxgen
%the sample distances: d 

[n,~]=size(X);
sizepop=options.pop;
alpha=options.alpha;
beta=options.beta;
p=options.p;
d=X; 
u=ceil(n/p);
maxgen=options.iterations;



initial_x=randn(n,p*sizepop);
initial_x(initial_x>=0)=1;
initial_x(initial_x<0)=-1;
initial_x=single(gpuArray(initial_x));
initial_v=single(gpuArray(zeros(n,p*sizepop)));





pbest=ones(sizepop,1)*inf;

C1=single(gpuArray(2));
C2=single(gpuArray(2));
y=zeros(maxgen,1);y=single(gpuArray(y));
ui=zeros(n,sizepop*p);ui=gpuArray(ui);




%obj=diag(half* x'*hat_W* x)'+theta_*  x;



initial_x=single(initial_x);

pbest=single(gpuArray(pbest));
d=single(gpuArray(d));


%initial_x=DiscreteHopfieldParallel1(d,beta,initial_x,n,P,hat_W,theta_array,iter,sizepop);
%[initial_x]=DiscreteHopfieldParallel(initial_x,ui, hat_W,theta_array,2000 ,sizepop);

I1=ones(n,n);      %the sum of each row
pop=sizepop;
I2=zeros(pop*p,pop*p); %the sum of each column
block=ones(p,p);
pbest_x=initial_x;
for i=1:pop
    I2((i-1)*p+1:p*i,(i-1)*p+1:p*i)=block;
end
I1=single(gpuArray(I1));
I2=single(gpuArray(I2));
stopNum=500;
m=zeros(n,sizepop*p);m=single(gpuArray(m));


one=single(gpuArray(eye(n,n)));
const1=0.5*d+beta*(I1-one);

one=single(gpuArray(eye(p*sizepop,p*sizepop)));
const2=alpha*(I2-one);
const3=repmat(alpha*(-2+p)+beta*(-2*u+n),n,p*sizepop);
zbest=inf;
ITER=600;
flag=single(gpuArray(ones(1,sizepop)));
for i = 1:maxgen
    [tx]=DiscreteHopfieldParallel11(m,initial_x,const1,const2,const3,ITER);
    constraint1=(tx*I2-2+p).*(tx*I2-2+p);constraint1=constraint1(:,1:p:end);constraint1=sum(constraint1,1);
    constraint2=(I1*tx-2*u+n).*(I1*tx-2*u+n);constraint2=constraint2(1,:);constraint2=reshape(constraint2,p,[]);
    constraint2=sum(constraint2,1);
    constraint=constraint1*alpha+constraint2*beta;
    % sum((sum(tx(:,(k-1)*p+1:p*k),2)-2+p).^2)*alpha+sum((sum(tx(:,(k-1)*p+1:p*k),1)-2*u+n).^2)*beta;
    obj=diag((tx'*(d-diag(diag(d)))*tx));obj=reshape(obj,p,[]);obj=(sum(obj,1)+constraint)*0.5;
    %局部最优
    for k=1:sizepop
        if obj(k)<pbest(k)
            pbest_x(:,(k-1)*p+1:p*k)=tx(:,(k-1)*p+1:p*k);
            pbest(k)=obj(k);
            flag(k)=constraint1(k);
        end
    end
    %全局最优
    [~,ids]=sort(pbest);
    best_id=ids(1);
    for k=1:sizepop
        if flag(ids(k))==0
            best_id=ids(k);
            break;
        end
    end
    if pbest(best_id)<zbest
    zbest_x = pbest_x(:,(best_id-1)*p+1:p*best_id);
    zbest=pbest(best_id);
    end
    
    rd1=single(gpuArray(rand(n,p*sizepop)));
    rd3=single(gpuArray(rand(n,p*sizepop)));
    initial_v=initial_v+C1*rd1.*((pbest_x + 1) / 2.0 - (initial_x + 1) / 2.0)+C2*rd1.*(repmat(zbest_x+1,1,sizepop) / 2.0 - (initial_x + 1) / 2.0);
    s=1./(1+exp(-initial_v));
    temp=s-rd3;
    temp(temp>=0)=1;
    temp(temp<0)=-1;
    initial_x=temp;
    y(i)=zbest;
    if stop(y,i,stopNum)
        break;
    end
end
%y=i-200;
y=gather(y);
l=label(gather(zbest_x),n);



%Parallel updating methods
function [x] = DiscreteHopfieldParallel(alpha,beta,u,n,p,sizepop,d,x,I1,I2,iter)
%parfor j=1:sizepop
%    uj=u(:,j);xj=x(:,j);thetaj=theta_(:,j);

%I=single(gpuArray(ones(1,size(x,1))));
%x=double(x);
%u=double(u);
%hat_W=double(hat_W);
%theta_=double(theta_);
%obj=[];


% const1=0.5*d+beta*I1;
% const2=alpha*I2;
% const3=repmat(alpha*(-2+p)+beta*(-2*u+n),n,p*sizepop);
for t=1:iter
    %0.5*t[pos] + ALPHA * (s1_i - x_pos - 2 + P) + BETA * (s2_k - x_pos - 2 * U + N);
    %dedx=0.5*d*x+alpha*(x*I2-2+p-x)+beta*(I1*x-2*u+n-x);
    dedx=const1*x+x*const2+const3;
    m = m-dedx;
%      uu=uu+abs(uu);
     x=2*sign(m+abs(m))-1;
%     x(m>=0)=1;
%     x(m<0)=-1;
%     uu=u+abs(u);
%     x=2*sign(uu)-1;
    %obj=[obj;diag(0.5* x'*hat_W* x)'+theta_(:,1)'*  x];
%      xjt=xj;
%      uj = uj-(hat_W*xj+thetaj);
%      uu = uj+abs(uj);
%      xj = sign(uu);
%      if isequal(xjt,xj)
%        break;
%      end
    %tx~=x(:,index);
   %resobj(t)=I*x(:,index);

%[pe1,pe2]=check(20,reshape(x(:,5),380,20),19,380);
end
%cc=1;
%x=single(x);

%    u(:,j)=uj;x(:,j)=xj;
    
%end
function [x] = DiscreteHopfieldParallel11(m,x,const1,const2,const3,iter)
%parfor j=1:sizepop
%    uj=u(:,j);xj=x(:,j);thetaj=theta_(:,j);

%I=single(gpuArray(ones(1,size(x,1))));
%x=double(x);
%u=double(u);
%hat_W=double(hat_W);
%theta_=double(theta_);
%obj=[];


% const1=0.5*d+beta*I1;
% const2=alpha*I2;
% const3=repmat(alpha*(-2+p)+beta*(-2*u+n),n,p*sizepop);
for t=1:iter
    %0.5*t[pos] + ALPHA * (s1_i - x_pos - 2 + P) + BETA * (s2_k - x_pos - 2 * U + N);
    %dedx=0.5*d*x+alpha*(x*I2-2+p-x)+beta*(I1*x-2*u+n-x);
    dedx=const1*x+x*const2+const3;
    m = m-dedx;
%      uu=uu+abs(uu);
     x=2*sign(m+abs(m))-1;
%     x(m>=0)=1;
%     x(m<0)=-1;
%     uu=u+abs(u);
%     x=2*sign(uu)-1;
    %obj=[obj;diag(0.5* x'*hat_W* x)'+theta_(:,1)'*  x];
%      xjt=xj;
%      uj = uj-(hat_W*xj+thetaj);
%      uu = uj+abs(uj);
%      xj = sign(uu);
%      if isequal(xjt,xj)
%        break;
%      end
    %tx~=x(:,index);
   %resobj(t)=I*x(:,index);

%[pe1,pe2]=check(20,reshape(x(:,5),380,20),19,380);
end
function [x]=DiscreteHopfieldSyn1(d,x,u,hat_W,theta_,iter,sizepop)
[m1,m2]=size(theta_);
n=144;
p=3;
u0=single(gpuArray(48));
alpha=500;
beta=500;
for i=1:m2
    y=x(:,i);y=reshape(y,144,3);
    %x(:,i)=DiscreteHopfieldSyn(y,n,d,p,u0,alpha,beta);
    x(:,i)=DiscreteHopfieldSingle(y,n,d,p,u0,alpha,beta);
    %x(:,i)=DiscreteHopfieldSyn(x(:,i),hat_W,theta_(:,i),iter,sizepop);
end
function [x]=DiscreteHopfieldSingle(x,n,d,p,u0,alpha,beta)
A=single(gpuArray(1));
B=single(gpuArray(alpha)); 
D=single(gpuArray(beta));
s1=single(gpuArray(sum(x,2)));
s2=single(gpuArray(sum(x,1)));
half=single(gpuArray(0.5));
t=single(gpuArray(zeros(n,p)));
for i=1:n
    for k=1:p
        t(i,k)=d(i,:)*x(:,k);
    end
end

for it=1:10
    rd=randperm(n*p);
    tx=x;
    for index=1:n*p
        [i,k]=ind2sub(size(x),rd(index));
        temp=x(i,k);
        dedx=A*t(i,k)+B*(s1(i)-half-x(i,k))+D*(s2(k)-x(i,k)-u0+half);
        u=-dedx;
        uu=u+abs(u);
        x(i,k)=sign(uu);
        s1(i)=s1(i)-temp+x(i,k);
        s2(k)=s2(k)-temp+x(i,k);
        t(:,k)=d(:,i)*x(i,k);
    end
    if isequal(x,tx)
        break;
    end
end

x=x(:);
function [x] = DiscreteHopfieldSyn(x,hat_W,theta_,iter,sizepop)
[m1,m2]=size(theta_);
for t=1:iter
    tx=x;
    rd=randperm(m1);
    for i=1:m1
        k=rd(i);
        u = -(hat_W(k,:)*x+theta_(k));
        uu=u +abs(u);
        x(k)=sign(uu);
    end
    if isequal(x,tx)
        break;
    end
    
end
%Constraints checking
function [pe1,pe2]=check(p,x,u,n)
[m,n]=size(x);
t2 = ones(1,m)*(x*ones(n,1)-1).^2;
t3=ones(1,p)*(x'*ones(m,1)-u);
pe1=t2;
pe2=t3;

%Stopping conditions
function flag=stop(y,i,k)
y=gather(y);
flag=0;
if i>=k
    if length(unique(y(i-k+1:i)))==1
        flag=1;
    end
end

function [x,iter] = DiscreteHopfieldParallel1(d,beta,x,n,p,hat_W,theta_,iter,sizepop)
%obj=single(gpuArray(zeros(100,500)));
[batch,relation,relation2]=GenerateBatch(n,p,sizepop);
relation=single(gpuArray(relation));
relation2=single(gpuArray(relation2));
batch=single(gpuArray(batch));
half=single(gpuArray(0.5));
n=single(gpuArray(n));
p=single(gpuArray(p));
np=n*p;
u0=single(gpuArray(1));
for it=1:10
    tx=x;

    for i=1:n
        index=mod(batch(i,:),np)+1;
        column=fix((index-1)/n)+1;
        %start=n*(column-1)+1;
        %finish=start+n-1;
        c1=relation(index,:);
        c2=relation2(index,:);
        dedx=beta*(sum((x(c1)),2)-u0+half)+sum(x(c2).*d(column,:),2);
         %dedx=A*t(i,k)+B*(s1(i)-half-x(i,k))+D*(s2(k)-x(i,k)-u0+half);
        u=-dedx;
        uu=u+abs(u);
        zz=sign(uu);
        x(batch(i,:)+1)=zz;
    end
 
    if isequal(tx,x)
        break;
    end
    %obj(it,:)=diag(0.5* x'*hat_W* x)'+theta_(:,1)'*  x;
end
function  x=plabel(l,p)
%鏍规嵁鏍囩姹傚嚭x
m=length(l);
x=zeros(m,p);
for i=1:m
    x(i,l(i))=1;
end