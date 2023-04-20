% 清空环境变量
clear
%% 网络结构建立
%读取数据
% data = load('flower.txt');
data = load('3.txt');
tic
[data,ps]=mapminmax(data',-1,1);
data=data';
%节点个数
inSize = 7;
outSize = 7;

%% 遗传算法参数初始化
maxgen=50;                         %进化代数，即迭代次数
sizepop=10;                        %种群规模
pcross=[0.4];                       %交叉概率选择，0和1之间
pmutation=[0.2];                    %变异概率选择，0和1之间

%节点总数
% numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;
numsum=2;
lenchrom=ones(1,numsum);                       
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
    individuals.fitness(i)=F(x,inSize,outSize,data);   %染色体的适应度
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
        individuals.fitness(j)=F(x,inSize,outSize,data);   
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
initial=round(0.34*size(data,1));%其中前cleanout数据用于初始化网络
Win = -0.5+rand(resSize,inSize);
W = -0.5+rand(resSize,resSize);
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
W = W .* (SP/rhoW);
Size_data=size(data,1);
% allocated memory for the design (collected states) matrix
X = zeros(inSize+resSize,initial-cleanout);

% set the corresponding target matrix directly
YTtest_T = data(initial+1:Size_data,:)';
YTrain_initial = data(cleanout+1:initial,:)';
% Y_Test=zeros();
% run the reservoir with the data and collect X
x = zeros(resSize,1);
for t = 1:initial
    u = data(t,:);
    u=u';
    x = tanh( Win*u + W*x );
    if t > cleanout
        X(:,t-cleanout) = [u;x];
    end
end

% train the output
XX(:,1:initial-cleanout)=X;
M=pinv(X * X');
beta = YTrain_initial*pinv(X);

%%%%%%%在线测试
for t = initial+1:Size_data-1
    u = data(t,:);
    u=u';
    x = tanh( Win*u + W*x );
    y = beta*[u;x];   
    z=[u;x];
    M = M - ((M*z*z'*M)/(1+z'*M*z));
    beta = beta + (data(t+1,:)' - beta*z)*z'*M;
    Y_Test(:,t-initial+1) = y;
end

MSE = mse(YTtest_T,Y_Test);
MAE=mae(YTtest_T,Y_Test);
RMSE=sqrt(MSE);
ave=mean(YTtest_T,'all');
NRMSE=RMSE/ave;
% nrmse = sqrt(sum((Y_Test-YTtest_T).^2)./sum((YTtest_T-ave).^2));
R2=1-mse(YTtest_T,Y_Test)/var(YTtest_T,0,'all');
disp( ['MSE = ', num2str( MSE )] );
disp( ['MAE = ', num2str( MAE )] );
disp( ['NRMSE = ', num2str( NRMSE )] );
disp( ['R2 = ', num2str( R2 )] );
disp( ['reSize = ', num2str( resSize )] );
disp( ['SP = ', num2str( SP )] );
Y_Test=Y_Test';
YTtest_T=YTtest_T';
figure(1);
plot(Y_Test(:,1));
hold on;
plot( YTtest_T(:,1) );
hold off;
axis tight;
title('Target and generated signals y(n) starting at n=0');
legend('Target signal', 'Free-running predicted signal');

figure(2);
plot(Y_Test(:,2));
hold on;
plot( YTtest_T(:,2) );
hold off;
axis tight;
title('Target and generated signals y(n) starting at n=0');
legend('Target signal', 'Free-running predicted signal');

figure(3);
plot(Y_Test(:,3));
hold on;
plot( YTtest_T(:,3) );
hold off;
axis tight;
title('Target and generated signals y(n) starting at n=0');
legend('Target signal', 'Free-running predicted signal');



figure(4);
plot(Y_Test(:,4));
hold on;
plot( YTtest_T(:,4) );
hold off;
axis tight;
title('Target and generated signals y(n) starting at n=0');
legend('Target signal', 'Free-running predicted signal');


figure(5);
plot(Y_Test(:,5));
hold on;
plot( YTtest_T(:,5) );
hold off;
axis tight;
title('Target and generated signals y(n) starting at n=0');
legend('Target signal', 'Free-running predicted signal');



figure(6);
plot(Y_Test(:,6));
hold on;
plot( YTtest_T(:,6) );
hold off;
axis tight;
title('Target and generated signals y(n) starting at n=0');
legend('Target signal', 'Free-running predicted signal');



figure(7);
plot(Y_Test(:,7));
hold on;
plot( YTtest_T(:,7) );
hold off;
axis tight;
title('Target and generated signals y(n) starting at n=0');
legend('Target signal', 'Free-running predicted signal');


figure(8);
plot(YTtest_T,'y' );
hold on;
plot(Y_Test,'b');
hold off;
axis tight;
title('Target and generated signals y(n) starting at n=0');
legend('Target signal', 'Free-running predicted signal');


Error1=YTtest_T(:,1)-Y_Test(:,1);
Error2=YTtest_T(:,2)-Y_Test(:,2);
Error3=YTtest_T(:,3)-Y_Test(:,3);
Error4=YTtest_T(:,4)-Y_Test(:,4);
Error5=YTtest_T(:,5)-Y_Test(:,5);
Error6=YTtest_T(:,6)-Y_Test(:,6);
Error7=YTtest_T(:,7)-Y_Test(:,7);
Mean_Error1=abs(mean(Error1))
Mean_Error2=abs(mean(Error2))
Mean_Error3=abs(mean(Error3))
Mean_Error4=abs(mean(Error4))
Mean_Error5=abs(mean(Error5))
Mean_Error6=abs(mean(Error6))
Mean_Error7=abs(mean(Error7))


Abs_Error1=abs((Error1));
Abs_Error2=abs((Error2));
Abs_Error3=abs((Error3));
Abs_Error4=abs((Error4));
Abs_Error5=abs((Error5));
Abs_Error6=abs((Error6));
Abs_Error7=abs((Error7));


figure(12);
plot(Mean_Error1);
hold on;
plot(Mean_Error2);
hold on;
plot(Mean_Error3);
hold on;
plot(Mean_Error4);
hold on;
plot(Mean_Error5);
hold on;
plot(Mean_Error6);
hold on;
plot(Mean_Error7);
hold off;
axis tight;
title('Error');


figure(15);
semilogy(Abs_Error1);
hold on;
semilogy(Abs_Error2);
hold on;
semilogy(Abs_Error3);
hold on;
semilogy(Abs_Error4);
hold on;
semilogy(Abs_Error5);
hold on;
semilogy(Abs_Error6);
hold on;
semilogy(Abs_Error7);
hold off;
axis tight;
title('Error');

figure(13);
plot(Error1(2:end));
hold on;
plot(Error2(2:end));
hold on;
plot(Error3(2:end));
hold on;
plot(Error4(2:end));
hold on;
plot(Error5(2:end));
hold on;
plot(Error6(2:end));
hold on;
plot(Error7(2:end));
hold off;
axis tight;
title('Error');
