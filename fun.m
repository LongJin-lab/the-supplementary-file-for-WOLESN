function error = fun(x,inSize,outSize,data)

resSize=round(x(1,1));
SP=x(1,2);

cleanout=100;
initial=500;
Block=500;
TrainingData=round(0.8*numel(data));
testLen=numel(data)-TrainingData;
Win = -0.5+rand(resSize,inSize);
W = -0.5+rand(resSize,resSize);
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
W = W .* ( SP /rhoW);
X = zeros(inSize+resSize,initial-cleanout);
XX=zeros(inSize+resSize,TrainingData-cleanout);
YTrain_initial = data(cleanout+1:initial)';
YTtest_T = data(TrainingData+1:TrainingData+testLen)';

x = zeros(resSize,1);
for t = 1:initial
    u = data(t);
    x = tanh( Win*[u] + W*x );
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
        u = data(t);
        Xb = zeros(inSize+resSize,size(Pn,1));
        x = tanh( Win*[u] + W*x );
        Xb(:,t-n+1) = [u;x];        
    end
    XX(:,initial-cleanout+1+(j-1)*size(Pn,1):initial-cleanout+(j)*size(Pn,1))=Xb;
    Xb=Xb';
   
    M = M - M * (Xb)' * (eye(Block) + Xb * M * (Xb)')^(-1) * Xb * M;
    beta = beta + M * (Xb)'* (Tn - Xb * beta);
end

Y_Test = zeros(outSize,testLen);
u = data(TrainingData+1);
for t = 1:testLen
    x = tanh( Win*[u] + W*x );
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

error=mse;