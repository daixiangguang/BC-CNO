%% I. 参数初始化
clc
clear
close all
load('iris.mat');
p=3;
fea=normlizedata(fea,2);
%fea=zscore(fea);
[n,m]=size(fea);
options.ReducedDim=fix(0.9*m);
W = PCA(fea,options);
fea=fea*W;
[n,~]=size(fea);

%QAPPSOHopfiledSyn 

options1.cores=1;
%options1.N=1;
options1.alpha=single(gpuArray(5));
options1.beta=single(gpuArray(5));
options1.iterations=20000;
options1.p=p;
   d=gaussinKernel(fea,0.08);
   d=-d;
   d=d-diag(diag(d));
   fea=d;




%修改以下代码
options1.N=1; %种群
pop=options1.N;
M=[10 20 30 50];
if pop==1
    y1=zeros(1,10);
else
    y1=zeros(4,10);
end

for it=1:10
    it
    y=QAPPSOHopfieldPal(fea,options1,200);
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

