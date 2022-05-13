function [batch_array]=GenerateBatch(n,p,sizepop)
batch=zeros(n,p);
block=zeros(1,p);
batch_array=zeros(n,p*sizepop);

for i=1:n
    s=i;
    for k=1:p
        block(k)=sub2ind([n,p],mod(s-1,n)+1,k);
        s=s+1;
    end
    batch(i,:)=block;
end


block=zeros(n,p);
for i=1:sizepop
    index=randperm(n);
    %index=1:n;
    for j=1:n
        block(j,:)=batch(index(j),:)+n*p*(i-1);
    end
    batch_array(:,(i-1)*p+1:i*p)=block;
end



% batch=(1:n*p)';
% block=zeros(n*p,1);
% for i=1:sizepop
%     index=randperm(n*p);
%     %index=1:n;
%     for j=1:n*p
%         block(j,:)=batch(index(j),:)+n*p*(i-1);
%     end
%     batch_array(:,i)=block;
% end

