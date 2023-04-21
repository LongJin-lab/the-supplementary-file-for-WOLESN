function ret=select(individuals,sizepop)

fitness1=10./individuals.fitness;   
sumfitness=sum(fitness1);
sumf=fitness1./sumfitness;

index=[]; 
for i=1:sizepop  
    pick=rand;
    while pick==0    
        pick=rand;        
    end
    for j=1:sizepop    
        pick=pick-sumf(j);        
        if pick<0        
            index=[index j];            
            break;  
        end
    end
end


individuals.chrom=individuals.chrom(index,:);
individuals.fitness=individuals.fitness(index);
ret=individuals;
 