clc
clear
load('wine.mat');
if ~exist('label','var')
    label=gnd;
end
nClass=max(unique(label));
fea=normlizedata(fea,2);
%parpool(64);
% fea=mapminmax(fea,0,1);
fea=zscore(fea);
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
d=gaussinKernel(fea,2);
d=-d;
d=d-diag(diag(d));

temp=0;
tic;
% y=cell(k,1);
pop=32;
p=max(label);
initialx=zeros(n*p,pop);
for i=1:pop
    temp=randn(n*p,1);
    temp(temp>=0)=1;
    temp(temp<0)=-1;
    initialx(:,i)=temp;
end
initialx=initialx';
i=8; % 1 4 8 16 32

M=10:200;
if pop==1
    y1=zeros(1,10);
else
    y1=zeros(length(M),10);
end
for it=1:10
    [y,x]=kernel(initialx,400);
    y=y(y~=0);
    if pop==1
        
    y1(1,it)=find_unstop_point(y,1);
    else
        for k=1:length(M)
    y1(k,it)=find_unstop_point(y,M(k));
        end
    end
end
yt=y1';

boxplot(yt(:,[5 25 45 65]),'Whisker',150)






