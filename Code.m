function ret=Code(lenchrom,bound)

flag=0;
while flag==0
    pick=rand(1,length(lenchrom));
    ret=bound(:,1)'+(bound(:,2)-bound(:,1))'.*pick;
    flag=test(lenchrom,bound,ret);     
end
        
