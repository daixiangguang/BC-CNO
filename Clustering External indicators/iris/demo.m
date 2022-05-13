clc
clear
load('iris.mat');
if ~exist('label','var')
    label=gnd;
end
nClass=max(unique(label));
fea=normlizedata(fea,2);
%parpool(64);
% fea=mapminmax(fea,0,1);
% fea=zscore(fea);
[n,m]=size(fea);
options.ReducedDim=fix(0.9*m);
W = PCA(fea,options);
[n,~]=size(fea);
fea=fea*W;
lambda=10;
beta=0.1;
gamma=0.1;
iter=100;
k=20;

% QAP
    d=gaussinKernel(fea,0.08);
    d=-d;
    d1=d;
    d=d-diag(diag(d));

QAP=zeros(k,3);
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



QAP_labels={};
tic;
for i=1:20
    i
    [y,x]=kernel(initialx);
    y=y(y~=0);
    l=mat2label(x',n);
    [NMI,AC]=ACNMI(l,label);
    QAP(i,1)=NMI;
    QAP(i,2)=AC;
     QAP(i,3)=y(end);
     QAP_labels{i}=l;
end
time=toc;
temp=temp(temp~=0);
temp=temp/k;
x_QAP=QAP;

NMI_QAP_avg=mean(x_QAP(:,1)*100);
NMI_QAP_max=max(x_QAP(:,1)*100);
NMI_QAP_min=min(x_QAP(:,1)*100);
NMI_QAP_std=std(x_QAP(:,1)*100);
AC_QAP_avg=mean(x_QAP(:,2)*100);
AC_QAP_max=max(x_QAP(:,2)*100);
AC_QAP_min=min(x_QAP(:,2)*100);
AC_QAP_std=std(x_QAP(:,2)*100);




