%% I. 参数初始化
clc
clear
close all
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
   d1=d;
   d=d-diag(diag(d));
QAP=zeros(k,3);
  fea1=d;
k=1;
x_result=zeros(k,2);

%QAPPSOHopfiledSyn 
options1.cores=1;
options1.pop=32;
options1.alpha=single(gpuArray(2));
options1.beta=single(gpuArray(2));
options1.iterations=20000;
options1.p=3;
QAPM_labels={};
for i=1:20
    i
    [l1,y1]=QAPPSOHopfieldPal(fea1,options1);
%     [S1,class1]=objective(fea,l1,n);
    [NMI1,AC1]=ACNMI(l1,label);
    x_result(i,:)=[NMI1,AC1];
    QAPM_labels{i}=l1;
end
NMI1_max=max(x_result(:,1));
NMI1_min=min(x_result(:,1));
NMI1_avg=mean(x_result(:,1));
AC1_max=max(x_result(:,2));
AC1_min=min(x_result(:,2));
AC1_avg=mean(x_result(:,2));
save QAPM_labels_wine.mat d1 fea QAPM_labels

