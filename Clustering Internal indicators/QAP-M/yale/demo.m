%% I. 参数初始化
clc
clear
load('Yale.mat');
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
    d=gaussinKernel(fea,4);
    d=-d;
    d1=d;
    d=d-diag(diag(d));
    fea1=d;
%QAPPSOHopfiledSyn 
options1.cores=1;
options1.pop=32;
options1.alpha=single(gpuArray(15));
options1.beta=single(gpuArray(15));
options1.iterations=20000;
options1.p=15;
QAPM_labels={};
for i=1:20
    i
    [l1,y1]=QAPPSOHopfieldPal(fea1,options1);
%     [S1,class1]=objective(fea,l1,n);
    [NMI1,AC1]=ACNMI(l1,label);
    x_result(i,:)=[NMI1,AC1];
    QAPM_labels{i}=l1;
end
NMI1_max=max(x_result(:,1)*100);
NMI1_min=min(x_result(:,1)*100);
NMI1_avg=mean(x_result(:,1)*100);
NMI1_std=std(x_result(:,1)*100);
AC1_max=max(x_result(:,2)*100);
AC1_min=min(x_result(:,2)*100);
AC1_avg=mean(x_result(:,2)*100);
AC1_std=std(x_result(:,2)*100);
save QAPM_labels_yale.mat d1 fea QAPM_labels
save result.mat x_result

