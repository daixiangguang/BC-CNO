function [l,y]=QAPPSOHopfieldSyn(x,options)
%Solving QAP by PSO and DHNN in the synchronous mode
%All codes are implemented on GPU
%the number of cpu cores:cores
%the number of population in each cpu core; pop
%the population size: sizepop=cores*pop
%the iteration counts: maxgen
%the sample distances: d 
[n,~]=size(x);
cores=options.cores;
pop=options.pop;
alpha=options.alpha;
beta=options.beta;
p=options.p;
iterations=options.iterations;
d=squareform(pdist(x)); 
u=ceil(n/p);
maxgen =options.iterations;   
sizepop = cores*pop;
[hat_W,theta_]=getCQPParameters(d,n,p,u,alpha,beta);
initial_x=gpuArray(zeros(n*p,sizepop));
initial_v=gpuArray(zeros(n*p,sizepop));


options.c1e=0.05;
options.c2e=0.2;
options.c1s=0.2;
options.c2s=0.05;
c1e=options.c1e;
c2e=options.c2e;
c1s=options.c1s;
c2s=options.c2s;
for i=1:sizepop
    temp=gpuArray(randi([0 1],n,p));
    initial_x(:,i)=temp(:);
end
initial_x=gpuArray(initial_x);

for i=1:sizepop
    iv=gpuArray(randn(n,p));
    initial_v(:,i)=iv(:);
end

pbest=ones(sizepop,1)*inf;
pbest_x=zeros(n*p,sizepop);

half=gpuArray(0.5);
y=zeros(maxgen,1);y=gpuArray(y);
theta_array=repmat(theta_',1,pop);theta_array=gpuArray(theta_array);theta_=gpuArray(theta_);
[m1,~]=size(theta_array);

hat_W=gpuArray(hat_W);
pbest=gpuArray(pbest);
pbest_x=gpuArray(pbest_x);


initial_x=single(initial_x);
hat_W=single(hat_W);

pbest=single(pbest);
pbest_x=single(pbest_x);

theta_=single(theta_);
half=single(half);

d=single(gpuArray(d));
u=single(gpuArray(u));

[batch]=GenerateBatch(n,p,sizepop);
options.d=d;
options.beta=beta;
options.x=d;
options.u0=u;
options.batch=batch;

options.p=p;
options.sizepop=sizepop;
options.n=n;

index=single((zeros(n,p*sizepop)));
row=single((zeros(p*sizepop,n)));
for i=1:n
    index(i,:)=mod(batch(i,:),n*p)+1;
    row(:,i)=mod((index(i,:)-1),n)+1;
end
options.index=index;
options.row=row;
%initial_x = Syn(initial_x,options);

for i = 1:iterations
    tx=initial_x;
    %if i>1
        [tx] = Syn(tx,options);
    %end
    obj=diag(half* tx'*hat_W* tx)'+theta_*  tx;
    tt=[pbest obj'];tt=transpose(tt);
    [least,index]=min(tt);
    pbest=least';
    index=repmat(index,m1,1);
    pbest_x(index==2)=tx(index==2);
    [~,bestindex] = min(pbest);
    zbest_x = pbest_x(:,bestindex);
    zbest=pbest(bestindex);
    [initial_x,initial_v]=ConPSO(initial_x,initial_v,pbest_x,zbest_x,1,0.05,0.05,sizepop);
   % c1=(c1e-c1s)*i/maxgen+c1s;
    %c2=(c2e-c2s)*i/maxgen+c2s;
   % initial_x=DiscretePSO(initial_x,pbest_x,zbest_x,n,p,sizepop,u,c1,c2);
    y(i)=zbest;
    if stop(y,i,200)
        break;
    end
end
y=gather(y);
l=label(reshape(gather(zbest_x),n,p),n);


function [x] = Syn(x,par)

d=par.d;
beta=par.beta;
p=par.p;
n=par.n;
u0=par.u0;
batch=par.batch;

sizepop=par.sizepop;
index=par.index;
row=par.row;
%obj=single(gpuArray(zeros(10,sizepop)));






one1=single(gpuArray(ones(n+p-2,1)));
one2=single(gpuArray(ones(n,1)));
flag=single(gpuArray(1));
it=1;




while(flag&&it<=20)
    tx=x;
    rd=single(randperm(n));
    for i=1:n
        id=index(rd(i),:);
        c1=relation1(id,:)+shift;
        c2=relation2(id,:)+shift;
        dedx=beta*(x(c1)*one1-u0)+x(c2).*d(row(:,i),:)*one2;
        %u=-dedx;
        %uu=u+abs(u);
        x(batch(rd(i),:)+1)=sign(abs(dedx)-dedx);
    end

    
    it=it+1;
    flag=~isequal(tx,x);
     %obj(it,:)=diag(0.5* x'*hat_W* x)'+theta_(1,:)*  x;
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
