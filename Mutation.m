function ret=Mutation(pmutation,lenchrom,chrom,sizepop,num,maxgen,bound)

for i=1:sizepop  
    pick=rand;
    while pick==0
        pick=rand;
    end
    index=ceil(pick*sizepop);
    
    pick=rand;
    if pick>pmutation
        continue;
    end
    flag=0;
    while flag==0
        pick=rand;
        while pick==0      
            pick=rand;
        end
        pos=ceil(pick*sum(lenchrom));      
        pick=rand;    
        fg=(rand*(1-num/maxgen))^2;
        if pick>0.5
            chrom(i,pos)=chrom(i,pos)+(bound(pos,2)-chrom(i,pos))*fg;
        else
            chrom(i,pos)=chrom(i,pos)-(chrom(i,pos)-bound(pos,1))*fg;
        end   
        flag=test(lenchrom,bound,chrom(i,:));     
    end
end
ret=chrom;