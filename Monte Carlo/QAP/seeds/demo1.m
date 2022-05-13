% clc
% clear
load('seeds.mat');
if ~exist('label','var')
    label=gnd;
end
nClass=max(unique(label));
fea=normlizedata(fea,2);
%parpool(64);
% fea=mapminmax(fea,0,1);
%fea=zscore(fea);
[n,m]=size(fea);
options.ReducedDim=fix(0.9*m);
W = PCA(fea,options);
[n,~]=size(fea);
fea=fea*W;
lambda=10;
beta=0.1;
gamma=0.1;
iter=100;
k=10;
% QAP
d=gaussinKernel(fea,0.05);
d=-d;
d=d-diag(diag(d));

temp=0;
tic;
% y=cell(k,1);
pop=32;
p=max(label);
initialx=zeros(n*p,pop);
rng(1);
for i=1:pop
    temp=randn(n*p,1);
    temp(temp>=0)=1;
    temp(temp<0)=-1;
    initialx(:,i)=temp;
end
initialx=initialx';
M=[5 25 45 65];
if pop==1
    y1=zeros(1,10);
else
    y1=zeros(4,10);
end
for it=1:10
    [y,x]=kernel(initialx,200);
    y=y(y~=0);
    if pop==1
        
    y1(1,it)=find_unstop_point(y,1);
    else
        
    y1(1,it)=find_unstop_point(y,M(1));
    y1(2,it)=find_unstop_point(y,M(2));
    y1(3,it)=find_unstop_point(y,M(3));
    y1(4,it)=find_unstop_point(y,M(4));
    end
end
   