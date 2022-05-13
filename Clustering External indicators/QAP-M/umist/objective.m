function [S,class]=objective(X,l,n)
% n表示样本个数
% k表示类
u=unique(l);%删除重复的标签
class=[];
for j=1:length(u)
    t=[];
    for i=1:n
        if l(i)==u(j)
            t=[t i];
        end
    end
    class=[class;t];
end
[c1,c2]=size(class);
S=0;
d=squareform(pdist(X)); 
for i=1:c1
    temp=0;
    for j=1:c2
        for o=j+1:c2
            index1=class(i,j);
            index2=class(i,o);
            temp=temp+d(index1,index2);
        end
    end
    S=S+temp;
end