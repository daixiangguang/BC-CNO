function l=label(x,n)
% try
l=zeros(n,1);
for i=1:n
    l(i)=find(x(i,:)==1);
end
% catch
%     pause
% end