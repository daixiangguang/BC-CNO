function [pbest,pbest_x]=getLocalBest(pbest,pbest_x,tx,hat_W,theta_,sizepop)
pbest=double(gather(pbest));
pbest_x=double(gather(pbest_x));
tx=double(gather(tx));
hat_W=double(gather(hat_W));
theta_=double(gather(theta_));
half=0.5;
obj=diag(half* tx'*hat_W* tx)'+theta_*  tx;
parfor j=1:sizepop
    if pbest(j)>obj(j)
        pbest(j)=obj(j);
        pbest_x(:,j)=tx(:,j);
    end
end
