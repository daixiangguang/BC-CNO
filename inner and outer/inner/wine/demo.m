
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
k=20;
% QAP
   d=gaussinKernel(fea,1);
  d=-d;
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
for i=1:1
    i
    
    [y,x]=kernel(initialx);
     y=y(y~=0);

end
ITER=10;
yb=y_processing(y,ITER);
plot(0:15,yb,'k');
hold on,
xlabel('Inner-loop Iteration','fontsize',15);
ylabel('$f_{\rho}(\underline{y})$','fontsize',15,'Interpreter','latex'); 