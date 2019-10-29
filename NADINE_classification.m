% MIT License
% Copyright (c) 2019
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:

% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

% The code/ implementation of adaptive memory strategy in NADINE adopts the
% concept of ellipsoidal anomaly detector implemented in the paper below:
% Masud Moshtaghi, James C. Bezdek, Christopher Leckie, Shanika Karunasekera,
% and Marimuthu Palaniswami. 2015. Evolving Fuzzy Rules for Anomaly Detection
% in Data Streams. IEEE Trans. Fuzzy Systems 23, 3 (2015), 688?700.
% https://doi. org/10.1109/TFUZZ.2014.2322385
% which code has been modified.

% Equation 1 : Network mean calculation in Line 441

% Equation 3 : Calculation of E[\hat{y}] in Line 412
% Equation 4 : Condition of hidden unit growing in Line 463
% Equation 5 : Condition of hidden unit pruning in Line 506
% Equation 6 : Complexity reduction strategy in Line 508
% Equation 7 : The condition of switching point (cut) in Line 194
% Equation 8 : The Calculation of the Hoeffding's error bound in Line  191
% and 192
% Equation 9 : The First Condition of sample to be stored in the adaptive
% memory. This method is implemented in Line 547
% Equation 10: The second condition of sample to be stacked in the adaptive
% memory implemented in Line 541
% Equation 11: The correlation of hidden layer to the target classes in
% Line 164
% Equation 12: Calculation of learning rate of hidden layer in Line 173


function [parameter,performance] = NADINE_classification(data,I,name)
%% divide the data into nFolds chunks
fprintf('========= NADINE algorithm is started=========\n')
tic
timeStampK = round(size(data,1)/1000);                                          % number of data chunk
[nData,mn] = size(data);
M = mn - I;
l = 0;
chunk_size = round(nData/timeStampK);
round_nFolds = floor(nData/chunk_size);
Data = {};
if round_nFolds == timeStampK
    if timeStampK == 1
        Data{1} = data;
    else
        for iHidLayer=1:timeStampK
            l=l+1;
            if iHidLayer < timeStampK
                Data1 = data(((iHidLayer-1)*chunk_size+1):iHidLayer*chunk_size,:);
            elseif iHidLayer == timeStampK
                Data1 = data(((iHidLayer-1)*chunk_size+1):end,:);
            end
            Data{l} = Data1;
        end
    end
else
    if timeStampK == 1
        Data{1} = data;
    else
        for iHidLayer=1:timeStampK-1
            l=l+1;
            Data1 = data(((iHidLayer-1)*chunk_size+1):iHidLayer*chunk_size,:);
            Data{l} = Data1;
        end
        iHidLayer = iHidLayer + 1;
        Data{timeStampK} = data(((iHidLayer-1)*chunk_size+1):end,:);
    end
end
clear data Data1

%% initiate model
K = 1;
parameter.networkConfiguration = 'stacked';    % stacked / parallel
parameter.net = netInit([I K M]);

%% initiate anomaly calculation
parameter.anomaly.Lambda              = 0.98;               % Forgetting factor
parameter.anomaly.StabilizationPeriod = 20;                 % The length of stabilization period.
parameter.anomaly.na                  = 10;                 % number of consequent anomalies to be considered as change
parameter.anomaly.Threshold1          = chi2inv(0.99 ,I);
parameter.anomaly.Threshold2          = chi2inv(0.999,I);
parameter.anomaly.firstinitdepth      = I + 1;
parameter.anomaly.indexkAnomaly       = 0;
parameter.anomaly.currentInvCovA      = 1*eye(I);
parameter.anomaly.currentMeanC        = zeros(1,I);
parameter.anomaly.CACounter           = 0;
parameter.anomaly.ChangePoints        = [];                 % Index of identified change points
parameter.anomaly.Anomaliesx          = [];                 % Identified anoamlies input
parameter.anomaly.AnomaliesT          = [];                 % Identified anoamlies target
parameter.net.outputConnect           = 0;                  % parallel connection

%% initiate node evolving iterative parameters
numHidLayer                 = 1;                            % number of initial hidden layer
parameter.ev{1}.hidlayer    = numHidLayer;
parameter.ev{1}.kp          = 0;
parameter.ev{1}.miu_x_old   = 0;
parameter.ev{1}.var_x_old   = 0;
parameter.ev{1}.kl          = 0;
parameter.ev{1}.K           = K;
parameter.ev{1}.node        = [];
parameter.ev{1}.BIAS2       = [];
parameter.ev{1}.VAR         = [];
parameter.ev{1}.miu_NS_old  = 0;
parameter.ev{1}.var_NS_old  = 0;
parameter.ev{1}.miu_NHS_old = 0;
parameter.ev{1}.var_NHS_old = 0;
parameter.ev{1}.miumin_NS   = [];
parameter.ev{1}.miumin_NHS  = [];
parameter.ev{1}.stdmin_NS   = [];
parameter.ev{1}.stdmin_NHS  = [];

%% initiate drift detection parameter
alpha_w = 0.0005;
alpha_d = 0.0001;
alpha   = 0.0001;


%% main loop, prequential evaluation
for k = 1:timeStampK
    %% load the data chunk-by-chunk
    network_evolhl(k,(1:parameter.net.nLayer-2))=parameter.net.initialConfig(1,2:end-1);
    numhl_evol(k)=parameter.net.nLayer-2;
    
    B = Data{k}(:,1:I);       % Data Stream Chunk
    Y = Data{k}(:,I+1:mn);    % Label
    clear Data{t}
    
    %% neural network testing
    fprintf('=========Chunk %d of %d=========\n', k, size(Data,2))
    disp('Discriminative Testing: running ...');
    parameter.net = testing(parameter.net,B,Y);
    LossFunct(k)=parameter.net.loss;
    parameter.ev{parameter.net.index}.t = k;
    clsRatePerBatch(k) = parameter.net.classRate;
    avgAccumClassificationRate(k) = mean(clsRatePerBatch);                          % Average of Classification Rate's Accummulation
    fprintf('Accummulated average classification rate : %d\n', avgAccumClassificationRate(k))
    disp('Discriminative Testing: ... finished');
    if k == timeStampK
        fprintf('========= NADINE has finished=========\n')
        break                   % last chunk only testing
    end
    
    %% Calculate correlation between hidden layers output and softmax output
    
    if numHidLayer > 1
        parameter.HnCorrOut = {};
        for iHidLayer = 1:numHidLayer
            for iHiddenNodes = 2:parameter.ev{iHidLayer}.K+1
                HnCorrOut = zeros(1,M);
                for iIndexInput = 1:M                                       % M ; number of input
                    temporary    = corrcoef(parameter.net.activity{iHidLayer + 1}(:,iHiddenNodes),parameter.net.activityOutput(:,iIndexInput));
                    HnCorrOut(iIndexInput) = abs(temporary(2,1));           %correlation of all nodes in the  hidden layer i to the output
                end
                parameter.HnCorrOut{iHidLayer}(iHiddenNodes-1) = mean(HnCorrOut);% correlation
            end
            SCorr(iHidLayer) = mean(parameter.HnCorrOut{iHidLayer});
        end    
        
        % Adaptive Learning Rate
        parameter.net.learningRate = zeros(1,numHidLayer);
        parameter.net.learningRate = 0.01*exp(-1*(1./SCorr-1));
        disp(parameter.net.learningRate);
        disp(SCorr);
        
    end
    
    %% Drift detection
    if k > 1
        cuttingpoint = 0;
        pp = size(Y,1);
        F_cut = zeros(pp,1);
        F_cut(parameter.net.wrongPred,:) = 1;
        [b_A_Upper,~] = max(F_cut);
        [b_A_lower,~] = min(F_cut);
        statistic_A = mean(F_cut);
        for cut = 1:pp
            statistic_B = mean(F_cut(1:cut,:));
            [a_A_Upper,~] = max(F_cut(1:cut,:));
            [a_A_Lower,~] = min(F_cut(1:cut,:));
            epsilon_B = (a_A_Upper - a_A_Lower)*sqrt(((pp)/(2*cut*(pp))*log(1/alpha)));
            epsilon_A = (b_A_Upper - b_A_lower)*sqrt(((pp)/(2*cut*(pp))*log(1/alpha)));
            if (epsilon_A + statistic_A)<=(epsilon_B + statistic_B)
                cuttingpoint = cut;
                statistic_H = mean(F_cut(cuttingpoint+1:end,:));
                epsilon_D = (b_A_Upper-b_A_lower)*sqrt(((pp - cuttingpoint)/(2*cuttingpoint*(pp - cuttingpoint)))  *log(1/alpha_d));
                epsilon_W = (b_A_Upper-b_A_lower)*sqrt(((pp - cuttingpoint)/(2*cuttingpoint*(pp - cuttingpoint)))*log(1/alpha_w));
                break
            end
        end
        if cuttingpoint == 0
            statistic_H = statistic_A;
            epsilon_D = (b_A_Upper-b_A_lower)*sqrt(((pp-cut)/(2*cut*(pp))*log(1/alpha_d)));
            epsilon_W = (b_A_Upper-b_A_lower)*sqrt(((pp-cut)/(2*cut*(pp))*log(1/alpha_w)));
        end
        if abs(statistic_B - statistic_H) > epsilon_D && cuttingpoint > 1 && cuttingpoint < pp
            st = 1;
            disp('Drift state: DRIFT');
            numHidLayer = numHidLayer + 1;
            parameter.net.nHiddenLayer = numHidLayer;
            parameter.net.nLayer = parameter.net.nLayer + 1;
            parameter.net.initialConfig(parameter.net.nHiddenLayer+1:parameter.net.nHiddenLayer+2)=[1 parameter.net.initialConfig(parameter.net.nHiddenLayer+1)];%update choi
            fprintf('The new Layer no %d is FORMED around chunk %d\n', numHidLayer, k)
            
            %% initiate NN weight parameters
            [ii,~] = size(parameter.net.weight{numHidLayer-1});
            parameter.net.weight{numHidLayer} = (rand(1,ii+1) - 0.5) * 2 * 4 * sqrt(6 / (M + ii));
            parameter.net.velocity{numHidLayer} = zeros(1,ii+1);
            parameter.net.grad{numHidLayer} = zeros(1,ii+1);
            
            %% initiate new classifier weight
            parameter.net.weightSoftmax = (rand(M,2) - 0.5) * 2 * 4 * sqrt(6 / (M + 2));
            parameter.net.velocitySoftmax = zeros(M,2);
            parameter.net.gradSoftmax = zeros(M,2);
            
            %% initiate iterative parameters
            parameter.ev{numHidLayer}.d       = numHidLayer;
            parameter.ev{numHidLayer}.kl          = 0;
            parameter.ev{numHidLayer}.K           = 1;
            parameter.ev{numHidLayer}.node        = [];
            parameter.ev{numHidLayer}.miu_NS_old  = 0;
            parameter.ev{numHidLayer}.var_NS_old  = 0;
            parameter.ev{numHidLayer}.miu_NHS_old = 0;
            parameter.ev{numHidLayer}.var_NHS_old = 0;
            parameter.ev{numHidLayer}.miumin_NS   = [];
            parameter.ev{numHidLayer}.miumin_NHS  = [];
            parameter.ev{numHidLayer}.stdmin_NS   = [];
            parameter.ev{numHidLayer}.stdmin_NHS  = [];
            parameter.ev{numHidLayer}.BIAS2       = [];
            parameter.ev{numHidLayer}.VAR         = [];
            
            %% check buffer
            if isempty(buffer_x)
                h = B;
                z = Y;
            else
                h = [buffer_x;B];
                z = [buffer_T;Y];
            end
            
            %% Constructing the input for next training
            
            h = [parameter.anomaly.Anomaliesx;h];
            h = [ones(size(h,1),1) h];
            parameter.net.activity{1} = h;
            Y = [parameter.anomaly.AnomaliesT;z];
            buffer_x = [];
            buffer_T = [];
            
            %% reset anomaly
            parameter.anomaly.indexkAnomaly = 0;
            parameter.anomaly.currentInvCovA = 1*eye(I);
            parameter.anomaly.currentMeanC = zeros(1,I);
            parameter.anomaly.CACounter = 0;
            parameter.anomaly.ChangePoints = [];                        % Index of identified change points
            
            
            
        elseif abs(statistic_B - statistic_H) >= epsilon_W && abs(statistic_B - statistic_H) < epsilon_D
            st = 2;
            disp('Drift state: WARNING');
            buffer_x = B;
            buffer_T = Y;
        else
            st = 3;
            disp('Drift state: STABLE');
            buffer_x = [];
            buffer_T = [];
        end
    else
        st = 3;
        disp('Drift state: STABLE');
        buffer_x = [];
        buffer_T = [];
    end
    drift(k) = st;
    HL(k) = numHidLayer;
    
    %% Discrinimanive training for all layers
    disp('Discriminative Training: running ...');
    parameter = training(parameter,Y,B);
    disp('Discriminative Training: ... finished');
    
    %% clear current chunk data
    clear Data{t}
    parameter.net.activity = {};
    fprintf('=========Hidden layers were updated=========\n')
    parameter.net.netpamsizeAll(k)=parameter.net.initialConfig_net_param;
end
toc

%Execution Time
exe_time = toc;
%Mean Classification Rate
r_cr = mean(clsRatePerBatch(2:end));
%Standard Deviation Size Evolution
r_std = std(clsRatePerBatch(2:end));
pamsizeevol=parameter.net.netpamsizeAll;


parameter.drift = drift;
performance.exe_time = exe_time;
performance.classification_rate = [mean(clsRatePerBatch(2:end)) std(clsRatePerBatch(2:end))];
performance.layer = [mean(HL) std(HL)];
performance.ParameterMeans=mean(pamsizeevol);
performance.ParameterStd=std(pamsizeevol);
performance.ClassificationRate=avgAccumClassificationRate;
performance.cr=clsRatePerBatch; % classification rate at time (t)
performance.LossFunct=LossFunct;
performance.HL=HL;
performance.network_evolhl=network_evolhl;
performance=generateFigure(name,performance);

% saving results on results folder
pathfolder=sprintf('results/%sresults',name);
namefile=sprintf('results/%sresults.mat',name);
saveas(gcf,pathfolder,'fig')
save(namefile,'avgAccumClassificationRate','clsRatePerBatch','r_std','pamsizeevol','exe_time','network_evolhl','numhl_evol','performance','HL','LossFunct');
end

function [miu] = meaniter(miu_old,x,k)
miu = miu_old + (x - miu_old)./k;
end

function p = probit(miu,std)
p = (miu./(1 + pi.*(std.^2)./8).^0.5);
end


function parameter  = training(parameter,y,B)
grow = 0;
prune = 0;

%% initiate performance matrix
ka = 0;
indexkAnomaly = parameter.anomaly.indexkAnomaly;
indexStableExecution = size(parameter.net.activity{1},2);

%% initiate performance matrix
indexHL          = parameter.net.nHiddenLayer;
kp          = parameter.ev{1}.kp;
miu_x_old   = parameter.ev{1}.miu_x_old;
var_x_old   = parameter.ev{1}.var_x_old;
kl          = parameter.ev{indexHL}.kl;
K           = parameter.ev{indexHL}.K;
node        = parameter.ev{indexHL}.node;
BIAS2       = parameter.ev{indexHL}.BIAS2;
VAR         = parameter.ev{indexHL}.VAR;
miu_NS_old  = parameter.ev{indexHL}.miu_NS_old;
var_NS_old  = parameter.ev{indexHL}.var_NS_old;
miu_NHS_old = parameter.ev{indexHL}.miu_NHS_old;
var_NHS_old = parameter.ev{indexHL}.var_NHS_old;
miumin_NS   = parameter.ev{indexHL}.miumin_NS;
miumin_NHS  = parameter.ev{indexHL}.miumin_NHS;
stdmin_NS   = parameter.ev{indexHL}.stdmin_NS;
stdmin_NHS  = parameter.ev{indexHL}.stdmin_NHS;

%% initiate training model
net                     = netInitTrain([parameter.net.initialConfig]);
net.activation_function = parameter.net.activation_function;
net.nLayer              = parameter.net.nLayer;
net.learningRate        = parameter.net.learningRate;

%% substitute the weight to be trained to training model
for lyr = 1:parameter.net.nLayer - 1
    if lyr + 1 == parameter.net.nLayer
        net.weight{lyr}     = parameter.net.weightSoftmax;
        net.velocity{lyr}   = parameter.net.velocitySoftmax;
        net.grad{lyr}       = parameter.net.gradSoftmax;
    else
        net.weight{lyr}     = parameter.net.weight{lyr};
        net.velocity{lyr}   = parameter.net.velocity{lyr};
        net.grad{lyr}       = parameter.net.grad{lyr};
    end
end
[~,bb] = size(parameter.net.weight{indexHL});

%% load the data for training
x     = parameter.net.activity{1};
[N,I] = size(x);
kk    = randperm(N);
x     = x(kk,:);
y     = y(kk,:);

%% main loop, train the model
for k   = 1 : N
    kp  = kp + 1;
    kl  = kl + 1;
    
    %% feedforward #1
    net     = netFeedForwardTrain(net, x(k,:), y(k,:));% the loss function is calculated here
    indexkAnomaly      = indexkAnomaly + 1;
    
    %% Incremental calculation of x_tail mean and variance
    [miu_x,std_x,var_x] = meanstditer(miu_x_old,var_x_old,net.activity{1},kp);
    miu_x_old           = miu_x;
    var_x_old           = var_x;
    
    %% Expectation of z
    Ey = probit(miu_x,std_x)'; % probit function is used (expectation is not linear)
    for iHiddenLayer = 1:parameter.net.nHiddenLayer
        py = sigmf(net.weight{iHiddenLayer}*Ey,[1,0]);
        py = [1;py];
        if iHiddenLayer == 1
            Ey2 = py.^2;
        end
        Ey=py;
    end
    clear Ey;
    Ey = py;
    Ez = net.weight{lyr}*Ey;
    Ez = exp(Ez);
    Ez = Ez./sum(Ez);
    
    if parameter.net.nHiddenLayer > 1
        py = Ey2;
        for iiHiddenLayer = 2:parameter.net.nHiddenLayer
            py = sigmf(net.weight{iiHiddenLayer}*py,[1,0]);
            py = [1;py];
        end
        Ey2 = py;
    end
    Ez2 = net.weight{lyr}*Ey2;
    Ez2 = exp(Ez2);
    Ez2 = Ez2./sum(Ez2);
    
    %% Network mean calculation
    bias2 = (Ez - y(k,:)').^2;
    ns = bias2;
    NS = norm(ns,'fro');
    
    %% Incremental calculation of NS mean and variance
    [miu_NS,std_NS,var_NS] = meanstditer(miu_NS_old,var_NS_old,NS,kp);
    miu_NS_old = miu_NS;
    var_NS_old = var_NS;
    miustd_NS = miu_NS + std_NS;
    
    a(k,:)=kl;
    
    if kl <= 1 || grow == 1
        miumin_NS = miu_NS;
        stdmin_NS = std_NS;
    else
        if miu_NS < miumin_NS
            miumin_NS = miu_NS;
        end
        if std_NS < stdmin_NS
            stdmin_NS = std_NS;
        end
    end
    miustdmin_NS = miumin_NS + (1.25*exp(-NS)+0.75)*stdmin_NS;
    BIAS2(kl,:)  = miu_NS;
    
    %% growing hidden unit
    if miustd_NS >= miustdmin_NS && kl > 1 % kl is the current data overtime
        grow = 1;
        K = K + 1;
        fprintf('The new node no %d is FORMED around sample %d\n', K, k)
        node(k+1) = K;net.size(parameter.net.nHiddenLayer+1)=K;
        %% Random Weight initialization
        net.weight{indexHL} = [net.weight{indexHL}; random('normal', 0, 0.01, 1, bb)];
        net.velocity{indexHL} = [net.velocity{indexHL};zeros(1,bb)];
        net.grad{indexHL} = [net.grad{indexHL};zeros(1,bb)];
        %% Random Weight initialization
        wNext = size(net.weight{indexHL+1},1);
        net.weight{indexHL+1} = [net.weight{indexHL+1} random('normal', 0, 0.01,wNext,1)];
        %%
        net.velocity{indexHL+1} = [net.velocity{indexHL+1} zeros(parameter.net.initialConfig(end),1)];
        net.grad{indexHL+1} = [net.grad{indexHL+1} zeros(parameter.net.initialConfig(end),1)];
    else
        grow = 0;
        node(k+1) = K;
    end
    %% Network variance calculation
    var = Ez2 - Ez.^2;
    NHS = norm(var,'fro');
    
    %% Incremental calculation of NHS mean and variance
    [miu_NHS,std_NHS,var_NHS] = meanstditer(miu_NHS_old,var_NHS_old,NHS,kp);
    miu_NHS_old = miu_NHS;
    var_NHS_old = var_NHS;
    miustd_NHS = miu_NHS + std_NHS;
    if kl <= I + 1 || prune == 1
        miumin_NHS = miu_NHS;
        stdmin_NHS = std_NHS;
    else
        if miu_NHS < miumin_NHS
            miumin_NHS = miu_NHS;
        end
        if std_NHS < stdmin_NHS
            stdmin_NHS = std_NHS;
        end
    end
    miustdmin_NHS = miumin_NHS + (2.5*exp(-NHS)+1.5)*stdmin_NHS;
    VAR(kl,:)     = miu_NHS;
    
    %% pruning hidden unit
    if grow == 0 && K > 1 && miustd_NHS >= miustdmin_NHS && kl > I + 1
        HS = Ey(2:end);
        [~,indexNode] = min(HS);
        fprintf('The node no %d is PRUNED around sample %d\n', indexNode, k)
        prune = 1;
        K = K - 1;
        node(k+1) = K; net.size(parameter.net.nHiddenLayer+1)=K;
        net.weight{indexHL}(indexNode,:) = [];
        net.velocity{indexHL}(indexNode,:) = [];
        net.grad{indexHL}(indexNode,:) = [];
        net.weight{indexHL+1}(:,indexNode+1) = [];
        net.velocity{indexHL+1}(:,indexNode+1) = [];
        net.grad{indexHL+1}(:,indexNode+1) = [];
    else
        node(k+1) = K;
        prune = 0;
    end
    
    if grow == 1 || prune == 1
        net = netFeedForwardTrain(net, x(k,:), y(k,:));
    end
    
    %% feedforward #2, executed if there is a hidden node changing
    net = lossBackward(net);
    net = optimizerStep(net);
    
    %% anomaly calculation
    if k <= size(B,1)
        if indexkAnomaly <= indexStableExecution
            parameter.anomaly.currentMeanC = meaniter(parameter.anomaly.currentMeanC,B(k,:),indexkAnomaly);% mean over time
        elseif indexkAnomaly > indexStableExecution
            mahaldist = (B(k,:) - parameter.anomaly.currentMeanC)*parameter.anomaly.currentInvCovA*(B(k,:) - parameter.anomaly.currentMeanC)';
            confCandidate = sort(net.activity{net.nLayer},'descend');
            y1 = confCandidate(1);
            y2 = confCandidate(2);
            confFinal = y1/(y1+y2);
            if (indexkAnomaly > parameter.anomaly.StabilizationPeriod)
                %Threshold 1 and Threshold 2 are obtained using chi2inv
                %(0.99,I) and chi2inv(0.999,I), the data point is regarded as an anomaly if
                % the condition below is fulfilled. After this condition is
                % executed, the CACounter is resetted to zero.
                if(mahaldist > parameter.anomaly.Threshold1 && mahaldist < parameter.anomaly.Threshold2) || confFinal <= 0.55 % check for anomalies
                    ka = ka + 1;
                    indexAnomaly(ka) = k;
                    parameter.anomaly.CACounter = 0;
                else
                    parameter.anomaly.CACounter = parameter.anomaly.CACounter + 1;
                end
            end
            if(parameter.anomaly.CACounter >= parameter.anomaly.na)
                parameter.anomaly.ChangePoints = [parameter.anomaly.ChangePoints;indexkAnomaly - parameter.anomaly.CACounter];
                parameter.anomaly.CACounter = 0;
            end
            [parameter.anomaly.currentInvCovA,parameter.anomaly.currentMeanC] = FormulatUpdate(parameter.anomaly.currentInvCovA,parameter.anomaly.currentMeanC,indexkAnomaly,parameter.anomaly.Lambda,B(k,:));
        end
    end
    
end

%% create buffer for anomaly
if ka ~= 0
    if size(parameter.anomaly.Anomaliesx,1) < 5000*indexHL
        parameter.anomaly.Anomaliesx = [parameter.anomaly.Anomaliesx;parameter.net.activity{1}(indexAnomaly,2:end)];
        parameter.anomaly.AnomaliesT = [parameter.anomaly.AnomaliesT;y(indexAnomaly,:)];
    elseif size(parameter.anomaly.Anomaliesx,1) >= 5000*indexHL
        n_anomaly = size(parameter.net.activity{1}(indexAnomaly,2:end),1);
        parameter.anomaly.Anomaliesx = [parameter.anomaly.Anomaliesx(n_anomaly+1:end,:);parameter.net.activity{1}(indexAnomaly,2:end)];
        parameter.anomaly.AnomaliesT = [parameter.anomaly.AnomaliesT(n_anomaly+1:end,:);y(indexAnomaly,:)];
    end
end

size_net_param=0;
for lyr = 1:parameter.net.nLayer - 1
    if lyr + 1 == parameter.net.nLayer
        parameter.net.weightSoftmax = net.weight{lyr};
        parameter.net.velocitySoftmax = net.velocity{lyr};
        parameter.net.gradSoftmax = net.grad{lyr};
    else
        parameter.net.weight{lyr} = net.weight{lyr};
        parameter.net.velocity{lyr} = net.velocity{lyr};
        parameter.net.grad{lyr} = net.grad{lyr};
    end
    if lyr>1 && lyr<parameter.net.nLayer
        size_net_param=size_net_param+numel(parameter.net.weight{lyr-1});
    end
end

size_net_param=size_net_param+numel(parameter.net.weightSoftmax);
parameter.net.initialConfig_net_param=size_net_param;

%% substitute the recursive calculation
parameter.ev{1}.kp           = kp;
parameter.ev{1}.miu_x_old    = miu_x_old;
parameter.ev{1}.var_x_old    = var_x_old;
parameter.ev{indexHL}.kl          = kl;
parameter.ev{indexHL}.K           = K;
parameter.ev{indexHL}.node        = node;
parameter.ev{indexHL}.BIAS2       = BIAS2;
parameter.ev{indexHL}.VAR         = VAR;
parameter.ev{indexHL}.miu_NS_old  = miu_NS_old;
parameter.ev{indexHL}.var_NS_old  = var_NS_old;
parameter.ev{indexHL}.miu_NHS_old = miu_NHS_old;
parameter.ev{indexHL}.var_NHS_old = var_NHS_old;
parameter.ev{indexHL}.miumin_NS   = miumin_NS;
parameter.ev{indexHL}.miumin_NHS  = miumin_NHS;
parameter.ev{indexHL}.stdmin_NS   = stdmin_NS;
parameter.ev{indexHL}.stdmin_NHS  = stdmin_NHS;
end


%% Initialize network for Training
function net = netInitTrain(layer)
net.initialConfig                       = layer;
net.nLayer                              = numel(net.initialConfig);
net.activation_function                 = 'relu';
net.learningRate                        = 0.01;
net.velocityCoeff                       = 0.95;
net.output                              = 'softmax';
end


%% feedforward
function net = testing(net, input, trueclass)

net                     = netFeedForward(net, input, trueclass);
[nData,~]               = size(trueclass);

%% obtain trueclass label
[~,actualLabel]         = max(trueclass,[],2);

%% obtain the class label
[~,net.classPrediction] = max(net.activityOutput,[],2);

%% calculate classification rate
net.wrongPred           = find(net.classPrediction ~= actualLabel);
net.classRate           = 1 - numel(net.wrongPred)/nData;
net.index               = net.nHiddenLayer;
end

function [miu,std,var] = meanstditer(miu_old,var_old,x,k)
miu = miu_old + (x - miu_old)./k;
var = var_old + (x - miu_old).*(x - miu);
std = sqrt(var/k);
end

function net = lossBackward(net)
nLayer = net.nLayer;
switch net.output
    case 'sigmf'
        backPropSignal{nLayer} = - net.error .* (net.activity{nLayer} .* (1 - net.activity{nLayer}));
    case {'softmax','linear'}
        backPropSignal{nLayer} = - net.error;          % loss derivative w.r.t. output
end

for iLayer = (nLayer - 1) : -1 : 2
    switch net.activation_function
        case 'sigmf'
            actFuncDerivative = net.activity{iLayer} .* (1 - net.activity{iLayer}); % contains b
        case 'tanh'
            actFuncDerivative = 1-net.activity{iLayer}.^2;
        case 'relu'
            actFuncDerivative = zeros(1,length(net.activity{iLayer}));
            actFuncDerivative(net.activity{iLayer}>0) = 0.1;
    end
    
    if iLayer+1 == nLayer
        backPropSignal{iLayer} = (backPropSignal{iLayer + 1} * net.weight{iLayer}) .* actFuncDerivative;
    else
        backPropSignal{iLayer} = (backPropSignal{iLayer + 1}(:,2:end) * net.weight{iLayer}) .* actFuncDerivative;
    end
end

for iLayer = 1 : (nLayer - 1)
    if iLayer + 1 == nLayer
        net.grad{iLayer} = (backPropSignal{iLayer + 1}' * net.activity{iLayer});
    else
        net.grad{iLayer} = (backPropSignal{iLayer + 1}(:,2:end)' * net.activity{iLayer});
    end
end
end

function net = netFeedForward(net, input, trueclass)
nLayer = net.nLayer;
nData = size(input,1);
input = [ones(nData,1) input];      % by adding 1 to the first coulomn, it means the first coulomn of weight is bias
net.activity{1} = input;            % the first activity is the input itself

%% feedforward from input layer through all the hidden layer
for iLayer = 2 : nLayer-1
    switch net.activation_function
        case 'sigmf'
            net.activity{iLayer} = sigmf(net.activity{iLayer - 1} * net.weight{iLayer - 1}',[1,0]);
        case 'relu'
            net.activity{iLayer} = max(net.activity{iLayer - 1} * net.weight{iLayer - 1}',0);
        case 'tanh'
            net.activity{iLayer} = tanh(net.activity{iLayer - 1} * net.weight{iLayer - 1}');
            
            
    end
    net.activity{iLayer} = [ones(nData,1) net.activity{iLayer}];
end

%% propagate to the output layer
switch net.output
    case 'sigmf'
        net.activityOutput = sigmf(net.activity{nLayer - 1} * net.weightSoftmax',[1,0]);
    case 'linear'
        net.activityOutput = net.activity{nLayer - 1} * net.weightSoftmax';
    case 'softmax'
        net.activityOutput = stableSoftmax(net.activity{nLayer - 1},net.weightSoftmax);
end

%% calculate error
net.error = trueclass - net.activityOutput;

%% calculate loss function
switch net.output
    case {'sigmf', 'linear'}
        net.loss = 1/2 * sum(sum(net.error .^ 2)) / nData;
    case 'softmax'
        net.loss = -sum(sum(trueclass .* log(net.activityOutput))) / nData;
end
end

function net = netFeedForwardTrain(net, input, trueClass)

nLayer = net.nLayer;
nData = size(input,1);
net.activity{1} = input;

%% feedforward from input layer through all the hidden layer
for iLayer = 2 : nLayer-1
    switch net.activation_function
        case 'sigmf'
            net.activity{iLayer} = sigmf(net.activity{iLayer - 1} * net.weight{iLayer - 1}',[1,0]);
        case 'relu'
            net.activity{iLayer} = max(net.activity{iLayer - 1} * net.weight{iLayer - 1}',0);
        case 'tanh'
            net.activity{iLayer} = tanh(net.activity{iLayer - 1} * net.weight{iLayer - 1}');
    end
    net.activity{iLayer} = [ones(nData,1) net.activity{iLayer}];
end

%% propagate to the output layer
switch net.output
    case 'sigmf'
        net.activity{nLayer} = sigmf(net.activity{nLayer - 1} * net.weight{nLayer - 1}',[1,0]);
    case 'linear'
        net.activity{nLayer} = net.activity{nLayer - 1} * net.weight{nLayer - 1}';
    case 'softmax'
        net.activity{nLayer} = stableSoftmax(net.activity{nLayer - 1},net.weight{nLayer - 1});
end
%% calculate error
net.error = trueClass - net.activity{nLayer};

%% calculate loss function
switch net.output
    case {'sigmf', 'linear'}
        net.loss = 1/2 * sum(sum(net.error .^ 2)) / nData;
    case 'softmax'
        net.loss = -sum(sum(trueClass .* log(net.activity{nLayer}))) / nData;
end
end




function net = optimizerStep(net)
lr = [net.learningRate 0.01];
if numel(lr) ~= net.nLayer - 1
    lr = [lr 0.01];
end
for iLayer = 1 : (net.nLayer - 1)
    if lr(iLayer) > 0
        grad                    = net.grad{iLayer};
        net.velocity{iLayer}    = net.velocityCoeff*net.velocity{iLayer} + lr(iLayer) * grad;
        finalGrad               = net.velocity{iLayer};
        net.weight{iLayer}      = net.weight{iLayer} - finalGrad;
    end
end
end


function [currentInvCovA,currentMeanC] = FormulatUpdate(PreviousMatrixA,previousMeanSample_k,indexOfSample, Lambda, currengDataSample)
default_Eff_Number = 200;
indexOfSample = min([indexOfSample default_Eff_Number]);
temp1 = (currengDataSample - previousMeanSample_k)*PreviousMatrixA*(currengDataSample - previousMeanSample_k)';
temp1 = temp1 + (indexOfSample - 1)/Lambda;
Mplier = ((indexOfSample)/((indexOfSample - 1)*Lambda));
currentInvCovA = PreviousMatrixA - ((PreviousMatrixA*(currengDataSample - previousMeanSample_k)'*(currengDataSample - previousMeanSample_k)*PreviousMatrixA)/temp1);
currentInvCovA = Mplier*currentInvCovA;
currentMeanC = Lambda*previousMeanSample_k + (1 - Lambda)*currengDataSample;
end


% This code aims to construct neural network with several hidden layer
% one can choose to either connect every hidden layer output to
% the last output or not

function net = netInit(layer)
net.initialConfig        = layer;                       % Initial network configuration
net.nLayer               = numel(net.initialConfig);    %  Number of layer
net.nHiddenLayer         = net.nLayer - 2;              %  Number of hidden layer
net.activation_function  = 'sigmf';                     %  Activation functions of hidden layers: 'sigmf', 'tanh' and 'relu'
net.learningRate         = 0.01;                        %  Learning rate Note: typically needs to be lower when using 'sigmf' activation function and non-normalized inputs.
net.velocityCoeff        = 0.95;                        %  Momentum coefficient, higher value is preferred
net.output               = 'softmax';                   %  output layer can be selected as follows: 'sigmf', 'softmax', and 'linear'

%% initiate weights and weight momentumCoeff for hidden layer
for i = 2 : net.nLayer - 1
    net.weight {i - 1}  = normrnd(0,1/sqrt(net.initialConfig(i - 1)),net.initialConfig(i),net.initialConfig(i - 1)+1);
    net.velocity{i - 1} = zeros(size(net.weight{i - 1}));
    net.grad{i - 1}     = zeros(size(net.weight{i - 1}));
    net.c{i - 1}        = rand(net.initialConfig(i - 1),1);
end

%% initiate weights and weight momentumCoeff for output layer
net.weightSoftmax   = normrnd(0,sqrt(2/(size(net.weight {i - 1},1)+1)),[net.initialConfig(end),net.initialConfig(end - 1)+1]);
net.velocitySoftmax = zeros(size(net.weightSoftmax));
net.gradSoftmax     = zeros(size(net.weightSoftmax));
end








function perform = generateFigure(name,performance)

ClassificationRate=performance.ClassificationRate;nFolds=length(ClassificationRate)+1;
cr=performance.cr;
LossFunct=performance.LossFunct;
HL=performance.HL;
network_evolhl=performance.network_evolhl;

subplot(4,2,1)
plot(ClassificationRate)
ylim([0.4 1]);
xlim([1 nFolds]);
ylabel({'Classification';'Rate'})
xlabel('Iteration/Chunk')
title({'a. The accummulated average of testing classification' sprintf('rate of %s dataset  over time', name) })

subplot(4,2,2)
plot(cr,'Color',[0 1 0])
ylim([0.4 1]);
xlim([1 nFolds]);
ylabel({'Classification';'Rate (t)'})
xlabel('Iteration/Chunk')
title({'b. The testing classification rate' sprintf('of %s dataset for each chunk', name)})

subplot(4,2,3)
plot(LossFunct,'Color',[1 0 0])
ylabel({'Discriminative' ;'Loss Function (t)'})
xlim([1 nFolds]);
xlabel('Iteration/Chunk');
title({sprintf('c. Discriminative Loss Function of %s for each chunk', name)})

subplot(4,2,4)

plot(HL,'Color',[0 0 1])
ylabel({'Number of'; 'Hidden Layer'})
xlim([1 nFolds]);
xlabel('Iteration/Chunk');
title({'d. The evolution of the number' ' of hidden layer in the network over time'})

subplot(4,1,[3 4])
plot(network_evolhl,'LineWidth',2,...
    'Color',[0.49 0.18 0.56])
ylabel('Number of hidden nodes')
xlim([1 nFolds]);
xlabel('Iteration/Chunk');
title({'e. The evolution of the number of hidden nodes in each layer in the network overtime'})
perform=performance;
end

function createfigure(Y1, Y2, Y3, Y4, YMatrix1)
%CREATEFIGURE(Y1, Y2, Y3, Y4, YMATRIX1)
%  Y1:  vector of y data
%  Y2:  vector of y data
%  Y3:  vector of y data
%  Y4:  vector of y data
%  YMATRIX1:  matrix of y data

%  Auto-generated by MATLAB on 27-Sep-2018 06:21:05

% Create figure
figure1 = figure;

% Create subplot
subplot1 = subplot(4,2,1,'Parent',figure1);
hold(subplot1,'on');

% Create plot
plot(Y1,'Parent',subplot1);

% Create xlabel
xlabel('Iteration/Chunk');

% Create ylabel
ylabel({'Classification','Rate'});

% Create title
title({'a. The accummulated average of testing classification ',' rate of SUSY dataset  over time'});

%% Uncomment the following line to preserve the X-limits of the axes
% xlim(subplot1,[1 2000]);
%% Uncomment the following line to preserve the Y-limits of the axes
% ylim(subplot1,[0.4 1]);
box(subplot1,'on');
% Create subplot
subplot2 = subplot(4,2,2,'Parent',figure1);
hold(subplot2,'on');

% Create plot
plot(Y2,'Parent',subplot2);

% Create xlabel
xlabel('Iteration/Chunk');

% Create ylabel
ylabel({'Classification','Rate (t)'});

% Create title
title({'b. The testing classification rate',' of SUSY dataset for each chunk'});

%% Uncomment the following line to preserve the X-limits of the axes
% xlim(subplot2,[1 2000]);
%% Uncomment the following line to preserve the Y-limits of the axes
% ylim(subplot2,[0.4 1]);
box(subplot2,'on');
% Create subplot
subplot3 = subplot(4,2,3,'Parent',figure1);
hold(subplot3,'on');

% Create plot
plot(Y3,'Parent',subplot3);

% Create xlabel
xlabel('Iteration/Chunk');

% Create ylabel
ylabel({'Discriminative','Loss Function'});

% Create title
title({'c. Discriminative Loss Function of SUSY for each chunk'});

%% Uncomment the following line to preserve the X-limits of the axes
% xlim(subplot3,[1 2000]);
box(subplot3,'on');
% Create subplot
subplot4 = subplot(4,2,4,'Parent',figure1);
hold(subplot4,'on');

% Create plot
plot(Y4,'Parent',subplot4);

% Create xlabel
xlabel('Iteration/Chunk');

% Create ylabel
ylabel({'Number of','Hidden Layer'});

% Create title
title({'d. The evolution of the number',' of hidden layer in the network over time'});

%% Uncomment the following line to preserve the X-limits of the axes
% xlim(subplot4,[1 2000]);
box(subplot4,'on');
% Create axes
axes1 = axes('Parent',figure1,...
    'Position',[0.13 0.11 0.775 0.376827956989247]);
hold(axes1,'on');

% Create multiple lines using matrix input to plot
plot(YMatrix1,'Parent',axes1);

% Create xlabel
xlabel('Iteration/Chunk');

% Create ylabel
ylabel('Number of Units');

% Create title
title({'e. The evolution of the number of hidden units in each layer in the network overtime'});

%% Uncomment the following line to preserve the X-limits of the axes
% xlim(axes1,[1 2000]);
box(axes1,'on');
end

%% stable softmax
function output = stableSoftmax(activation,weight)
output = activation * weight';
output = exp(output - max(output,[],2));
output = output./sum(output, 2);
end
