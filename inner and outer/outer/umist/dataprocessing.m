function yb=dataprocessing(y)
l=find(y==0);
[~,column]=ind2sub(size(y),l(1));
% yb=y(:,1:column-1);


tail=ceil(column*0.3);
yb=zeros(size(y,1)+1,column+tail-1);
yb(1:end-1,1:column-1)=y(:,1:column-1);
yb(1:end-1,column:end)=repmat(y(:,column-1),1,tail);

[row,column]=size(yb);
for i=1:column
    yb(end,i)=min(yb(:,i));
end

