% 清空环境变量
clear
data = load('MackeyGlass_t200.txt');
[data,ps]=mapminmax(data',-1,1);
data=data';

inSize = 1;
outSize = 1;

%% 遗传算法参数初始化
maxgen=50;                         
sizepop=10;                        
pcross=[0.4];                       
pmutation=[0.2];                   


numsum=2;
lenchrom=ones(1,numsum);                       
bound=zeros(2,2);
bound(1,1)=10;
bound(1,2)=250;
bound(2,1)=0;
bound(2,2)=1;

individuals=struct('fitness',zeros(1,sizepop), 'chrom',[]);  
avgfitness=[];                      
bestfitness=[];                    
bestchrom=[];                      

for i=1:sizepop
    individuals.chrom(i,:)=Code(lenchrom,bound);    
    x=individuals.chrom(i,:);    
    individuals.fitness(i)=fun(x,inSize,outSize,data);   
end
FitRecord=[];
[bestfitness bestindex]=min(individuals.fitness);
bestchrom=individuals.chrom(bestindex,:);  
avgfitness=sum(individuals.fitness)/sizepop; 
trace=[avgfitness bestfitness]; 

for i=1:maxgen  
    individuals=Select(individuals,sizepop); 
    avgfitness=sum(individuals.fitness)/sizepop;
    individuals.chrom=Cross(pcross,lenchrom,individuals.chrom,sizepop,bound);
    individuals.chrom=Mutation(pmutation,lenchrom,individuals.chrom,sizepop,i,maxgen,bound);
    
   
    for j=1:sizepop
        x=individuals.chrom(j,:);
        individuals.fitness(j)=fun(x,inSize,outSize,data);   
    end    
   
    [newbestfitness,newbestindex]=min(individuals.fitness);
    [worestfitness,worestindex]=max(individuals.fitness);
    
    if bestfitness>newbestfitness
        bestfitness=newbestfitness;
        bestchrom=individuals.chrom(newbestindex,:);
    end
    individuals.chrom(worestindex,:)=bestchrom;
    individuals.fitness(worestindex)=bestfitness;
   
    avgfitness=sum(individuals.fitness)/sizepop;
    trace=[trace;avgfitness bestfitness]; 
    FitRecord=[FitRecord;individuals.fitness];
end

resSize=round(x(1,1));
SP=x(1,2);

cleanout=100;
initial=500;
a = 1; 
Block=500;
TrainingData=round(0.8*numel(data));
testLen=numel(data)-TrainingData;
Win = -0.5+rand(resSize,inSize);
W = -0.5+rand(resSize,resSize);
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
W = W .* (SP/rhoW);

X = zeros(inSize+resSize,initial-cleanout);
XX=zeros(inSize+resSize,TrainingData-cleanout);
YTrain_initial = data(cleanout+1:initial)';
YTrain_T = data(cleanout+1:TrainingData)';
YTtest_T = data(TrainingData+1:TrainingData+testLen)';

x = zeros(resSize,1);
for t = 1:initial
    u = data(t);
    x = (1-a)*x + a*tanh( Win*[u] + W*x );
    if t > cleanout
        X(:,t-cleanout) = [u;x];
    end
end


X=X';
XX(:,1:initial-cleanout)=X';
M = pinv(X' * X); 
beta = pinv(X) * YTrain_initial';

j=0;
for n = initial : Block : TrainingData
    j=j+1;
    if (n+Block-1) > TrainingData
        Pn = data(n:TrainingData,:);
        Tn = data(n+1:TrainingData+1,:);
        Block = size(Pn,1);             
    else
        Pn = data(n:(n+Block-1),:);
        Tn = data(n+1:(n+Block-1)+1,:);
    end
    size(Pn,1);
    for t = n:n+size(Pn,1)-1
        aa=size(Pn,1);
        u = data(t);
        Xb = zeros(inSize+resSize,size(Pn,1));
        x = (1-a)*x + a*tanh( Win*[u] + W*x );
        Xb(:,t-n+1) = [u;x];
        
    end
    XX(:,initial-cleanout+1+(j-1)*size(Pn,1):initial-cleanout+(j)*size(Pn,1))=Xb;
    Xb=Xb';
    M = M - M * (Xb)' * (eye(Block) + Xb * M * (Xb)')^(-1) * Xb * M;
    beta = beta + M * (Xb)'* (Tn - Xb * beta);
end

Y_Train=XX' * beta;
Y_Test = zeros(outSize,testLen);
u = data(TrainingData+1);
for t = 1:testLen
    x = (1-a)*x + a*tanh( Win*[u] + W*x );
    y = beta'*[u;x];
    Y_Test(:,t) = y;
    if(TrainingData+t+1>size(data))
        break;
    else
        u = data(TrainingData+t+1);
    end
end

TestErrorLen = testLen;
mse = sum((Y_Test-YTtest_T).^2)./TestErrorLen;
ave=mean(YTtest_T);
nrmse = sqrt(sum((Y_Test-YTtest_T).^2)./sum((YTtest_T-ave).^2));
mae = mean(abs(Y_Test-YTtest_T));
disp( ['MSE = ', num2str( mse )] );
disp( ['MAE = ', num2str( mae )] );
disp( ['NRMSE = ', num2str( nrmse )] );
disp( ['reSize = ', num2str( resSize )] );
disp( ['SP = ', num2str( SP )] );
