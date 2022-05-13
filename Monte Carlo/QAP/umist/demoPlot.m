N=[1 16 32 64 96]; %种群个数
M=[10 50 100 200];%不动点个数
yb=combine(N);
boxplot(yb','Whisker',150);
% N=[1 4 8 16 32]; %种群个数
%  set(gca,'xticklabels',{ str11});%id=1
% 10 100 200 300
ylim([-108500,-102200]);
set(gca,'xticklabels',{ 'N=1','M=10', '50', '100', '200','M=10', '50', '100', '200','M=10', '50', '100', '200','M=10', '50', '100', '200','300'});%id=3

set(gca,'TickLabelInterpreter','latex');
set(gca, 'XTickLabelRotation', 45);% 横坐标倾斜45度
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);% 设置论文字体
xlabel('~\emph{N}=16~~~~~~~~~~~~~\emph{N}=32~~~~~~~~~~~~~\emph{N}=64~~~~~~~~~~~~~\emph{N}=96~~~~~~~','Interpreter','latex') 
ylabel('$f_{\rho}(\underline{y})$','FontSize',15,'Interpreter','latex')