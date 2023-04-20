% 清空环境变量
clear
%% 网络结构建立
%读取数据
data = load('MackeyGlass_t200.txt');
% data = load('JXB4.csv');
% data = load('flower.txt');
% data = load('test.txt');
% data = load('sunspot.txt');
% plot(data(1:500));
[data,ps]=mapminmax(data',-1,1);
data=data';
%节点个数
inSize = 1;
outSize = 1;

%% 遗传算法参数初始化
maxgen=50;                         %进化代数，即迭代次数
sizepop=10;                        %种群规模
pcross=[0.4];                       %交叉概率选择，0和1之间
pmutation=[0.2];                    %变异概率选择，0和1之间

%节点总数
% numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;
numsum=2;
lenchrom=ones(1,numsum);                       %个体长度
% bound=[-3*ones(numsum,1) 3*ones(numsum,1)];    %个体范围
bound=zeros(2,2);
bound(1,1)=10;
bound(1,2)=250;
bound(2,1)=0;
bound(2,2)=1;

individuals=struct('fitness',zeros(1,sizepop), 'chrom',[]);  %将种群信息定义为一个结构体
avgfitness=[];                      %每一代种群的平均适应度
bestfitness=[];                     %每一代种群的最佳适应度
bestchrom=[];                       %适应度最好的染色体
%计算个体适应度值
for i=1:sizepop
    %随机产生一个种群
    individuals.chrom(i,:)=Code(lenchrom,bound);    %编码（binary和grey的编码结果为一个实数，float的编码结果为一个实数向量）
    x=individuals.chrom(i,:);
    %计算适应度    
    individuals.fitness(i)=fun(x,inSize,outSize,data);   %染色体的适应度
end
FitRecord=[];
%找最好的染色体
[bestfitness bestindex]=min(individuals.fitness);
bestchrom=individuals.chrom(bestindex,:);  %最好的染色体
avgfitness=sum(individuals.fitness)/sizepop; %染色体的平均适应度
%记录每一代进化中最好的适应度和平均适应度
trace=[avgfitness bestfitness]; 

%% 迭代求解最佳初始阀值和权值
% 进化开始
for i=1:maxgen  
    % 选择
    individuals=Select(individuals,sizepop); 
    avgfitness=sum(individuals.fitness)/sizepop;
    %交叉
    individuals.chrom=Cross(pcross,lenchrom,individuals.chrom,sizepop,bound);
    % 变异
    individuals.chrom=Mutation(pmutation,lenchrom,individuals.chrom,sizepop,i,maxgen,bound);
    
    % 计算适应度 
    for j=1:sizepop
        x=individuals.chrom(j,:); %个体
        individuals.fitness(j)=fun(x,inSize,outSize,data);   
    end
    
    %找到最小和最大适应度的染色体及它们在种群中的位置
    [newbestfitness,newbestindex]=min(individuals.fitness);
    [worestfitness,worestindex]=max(individuals.fitness);
    
    %最优个体更新
    if bestfitness>newbestfitness
        bestfitness=newbestfitness;
        bestchrom=individuals.chrom(newbestindex,:);
    end
    individuals.chrom(worestindex,:)=bestchrom;
    individuals.fitness(worestindex)=bestfitness;
    
    %记录每一代进化中最好的适应度和平均适应度
    avgfitness=sum(individuals.fitness)/sizepop;
    trace=[trace;avgfitness bestfitness]; 
    FitRecord=[FitRecord;individuals.fitness];
end

%% 把最优存储池尺寸，谱半径赋予网络预测
% %用遗传算法优化的ESN网络进行值预测
resSize=round(x(1,1));
SP=x(1,2);
%网络训练
cleanout=100;
initial=500;%其中前cleanout数据用于初始化网络
a = 1; % leaking rate
Block=500;
TrainingData=round(0.8*numel(data));
testLen=numel(data)-TrainingData;
% rand( 'seed', 4555555555 );
Win = -0.5+rand(resSize,inSize);
W = -0.5+rand(resSize,resSize);
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
W = W .* (SP/rhoW);%从1开始向左右两边取值 添加选取参数的实验

% allocated memory for the design (collected states) matrix
X = zeros(inSize+resSize,initial-cleanout);
XX=zeros(inSize+resSize,TrainingData-cleanout);
% set the corresponding target matrix directly
YTrain_initial = data(cleanout+1:initial)';
YTrain_T = data(cleanout+1:TrainingData)';
YTtest_T = data(TrainingData+1:TrainingData+testLen)';

% run the reservoir with the data and collect X
x = zeros(resSize,1);
for t = 1:initial
    u = data(t);
    x = (1-a)*x + a*tanh( Win*[u] + W*x );
    if t > cleanout
        X(:,t-cleanout) = [u;x];
    end
end

% train the output
X=X';
XX(:,1:initial-cleanout)=X';
M = pinv(X' * X); %M=P P=(K)_1 K=H'*H
beta = pinv(X) * YTrain_initial';

%%%%%%%%%%%%% step 2 Sequential Learning Phase
j=0;
for n = initial : Block : TrainingData
    j=j+1;
    if (n+Block-1) > TrainingData
        Pn = data(n:TrainingData,:);
        Tn = data(n+1:TrainingData+1,:);
        Block = size(Pn,1);             %%%% correct the block size
        %%%% correct the first dimention of V
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
    %eye(n)返回一个主对角线元素为 1 且其他位置元素为 0 的 n×n 单位矩阵。
    M = M - M * (Xb)' * (eye(Block) + Xb * M * (Xb)')^(-1) * Xb * M;
    %     beta = beta + (Tn - beta * (Xb)')* (Xb) * M;
    beta = beta + M * (Xb)'* (Tn - Xb * beta);
end
%disp( ['time = ', num2str( toc )] );
%训练输出
Y_Train=XX' * beta;
% run the trained ESN in a generative mode. no need to initialize here,
% because x is initialized with training data and we continue from there.
Y_Test = zeros(outSize,testLen);
u = data(TrainingData+1);
for t = 1:testLen
    x = (1-a)*x + a*tanh( Win*[u] + W*x );
    y = beta'*[u;x];
    Y_Test(:,t) = y;
    % generative mode:
    % %     u = y;
    % this would be a predictive mode:
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
