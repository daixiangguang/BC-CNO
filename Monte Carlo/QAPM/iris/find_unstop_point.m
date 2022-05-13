function obj=find_unstop_point(y,M)
obj=0;
if M==1
    obj=y(1);
else
    i=1;
    count=0;
    unstop=y(i);
    for loop=i+1:length(y)
        if unstop==y(loop)
            count=count+1;
        else
            count=0;
            unstop=y(loop);
        end
        if count==M
            obj=unstop;
            break;
        end     
    end
end