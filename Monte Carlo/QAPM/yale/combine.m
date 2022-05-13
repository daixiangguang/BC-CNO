function yb=combine(N)
yb=[];
str='y_N=';
for i=1:length(N)
    load(strcat(str,num2str(N(i))));
    yb=[yb;y1];
end




