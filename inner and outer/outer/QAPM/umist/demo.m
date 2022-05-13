%% I. 参数初始化
clc
clear
close all
load('umist.mat');
if ~exist('label','var')
    label=gnd;
end
nClass=max(unique(label));
fea=normlizedata(fea,1);
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
k=10;
% QAP
   d=gaussinKernel(fea,0.2);
  d=-d;
   d=d-diag(diag(d));
QAP=zeros(k,3);
  fea=d;
k=1;
x_result=zeros(k,2);

%QAPPSOHopfiledSyn 
options1.cores=1;
options1.pop=32;
options1.alpha=single(gpuArray(15));
options1.beta=single(gpuArray(15));
options1.iterations=20000;
options1.p=20;

for i=1:1
    [objectives]=QAPPSOHopfieldPal(fea,options1);
end
ya=double(gather(objectives));
ya=ya';
yb=dataprocessing(ya);
yb=yb(:,1:4501);
[row,column]=size(yb);
for i=1:size(yb,1)
    if i~=row
        plot(0:4500,yb(i,:));
        hold on;
    else
        plot(0:4500,yb(i,:),'r');
        hold on;
    end
    xlabel('Outer-loop Iteration','fontsize',15);
    ylabel('$f_{\rho}(\underline{y})$','fontsize',15,'Interpreter','latex'); 
end