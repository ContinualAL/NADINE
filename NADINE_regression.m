% MIT License
%
% Copyright (c) 2019 Choiru Za'in Andri Ashfahani Mahardhika Pratama
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE. XAVIER

% The code/implementation of adaptive memory strategy in NADINE adopts the
% concept of ellipsoidal anomaly detector implemented in the paper below:
% Masud Moshtaghi, James C. Bezdek, Christopher Leckie, Shanika Karunasekera,
% and Marimuthu Palaniswami. 2015. Evolving Fuzzy Rules for Anomaly Detection
% in Data Streams. IEEE Trans. Fuzzy Systems 23, 3 (2015), 688?700.
% https://doi. org/10.1109/TFUZZ.2014.2322385
% which code has been modified.

% Equation 1 : Network mean calculation in Line 671

% Equation 3 : Calculation of E[\hat{y}] in Line 648
% Equation 4 : Condition of hidden unit growing in Line 694
% Equation 5 : Condition of hidden unit pruning in Line 735
% Equation 6 : Complexity reduction strategy in Line 736
% Equation 7 : The condition of switching point (cut) in Line 210
% Equation 8 : The Calculation of the Hoeffding's error bound in Line 207 
% and 208
% Equation 9 : The First Condition of sample to be stored in the adaptive
% memory. This method is implemented in Line 772
% Equation 10: The second condition of sample to be stacked in the adaptive
% memory implemented in Line 769
% Equation 11: The correlation of hidden layer to the target classes in
% Line 162-182
% Equation 12: Calculation of learning rate of hidden layer in Line 181



%% main code
function [parameter,performance] = NADINE_regression(data,I)
%% divide the data into nFolds chunks
fprintf('=========NADINE Learning is started=========\n')
nFolds = round(size(data,1)/1);                 % number of data chunk
[nData,mn] = size(data);
M = mn - I;
l = 0;
chunk_size = round(nData/nFolds);
round_nFolds = floor(nData/chunk_size);
Data = {};
if round_nFolds == nFolds
    if nFolds == 1
        Data{1} = data;
    else
        for iHidLayer=1:nFolds
            l=l+1;
            if iHidLayer < nFolds
                Data1 = data(((iHidLayer-1)*chunk_size+1):iHidLayer*chunk_size,:);
            elseif iHidLayer == nFolds
                Data1 = data(((iHidLayer-1)*chunk_size+1):end,:);
            end
            Data{l} = Data1;
        end
    end
else
    if nFolds == 1
        Data{1} = data;
    else
        for iHidLayer=1:nFolds-1
            l=l+1;
            Data1 = data(((iHidLayer-1)*chunk_size+1):iHidLayer*chunk_size,:);
            Data{l} = Data1;
        end
        iHidLayer = iHidLayer + 1;
        Data{nFolds} = data(((iHidLayer-1)*chunk_size+1):end,:);
    end
end
clear data Data1

%% initiate model
K = 1;
parameter.networkConfiguration = 'stacked';    % stacked / parallel
parameter.net = netInit([I K M]); parameter.net.output = 'linear';

%% initiate anomaly calculation
parameter.anomaly.Lambda              = 0.98;                   % Forgetting factor
parameter.anomaly.StabilizationPeriod = 20;                     % The length of stabilization period.
parameter.anomaly.na                  = 10;                     % number of consequent anomalies to be considered as change
parameter.anomaly.Threshold1          = chi2inv(0.99 ,I);
parameter.anomaly.Threshold2          = chi2inv(0.999,I);
parameter.anomaly.firstinitdepth      = I + 1;
parameter.anomaly.indexAnomaly        = 0;
parameter.anomaly.currentInvCovA      = 1*eye(I);
parameter.anomaly.currentMeanC        = zeros(1,I);
parameter.anomaly.CACounter           = 0;
parameter.anomaly.ChangePoints        = [];                     % Index of identified change points
parameter.anomaly.Anomaliesx          = [];                     % Identified anoamlies input
parameter.anomaly.AnomaliesT          = [];                     % Identified anoamlies target
parameter.net.outputConnect           = 0;                      % parallel connection

%% initiate node evolving iterative parameters
numHidLayer                 = 1;     % number of hidden layer
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
parameter.ev{1}.grow        = 0;
parameter.ev{1}.prune       = 0;

%% main loop, prequential evaluation
countF_cut = 1;
for t = 1:nFolds
    %% load the data chunk-by-chunk
    x = Data{t}(:,1:I);
    x_ori = x;
    T = Data{t}(:,I+1:mn);
    T_ori = T;
    [bd,~] = size(T);
    clear Data{t}
    
    %% neural network testing
    start_test = tic;
    fprintf('=========Chunk %d of %d=========\n', t, size(Data,2))
    disp('Discriminative Testing: running ...');
    parameter.net.t = t;
    
    net = testing(parameter.net,x,T,parameter.ev);
    parameter.net = net;
    
    actualOutput(bd*t+(1-bd):bd*t,:) = T;
    predictedOutput(bd*t+(1-bd):bd*t,:) = parameter.net.activityOutput;
    
    parameter.ev{parameter.net.index}.t = t;
    parameter.squareError(bd*t+(1-bd):bd*t,:) = parameter.net.squareError;
    
    disp('Discriminative Testing: ... finished');
    if t == nFolds
        fprintf('=========NADINE Learning is finished=========\n')
        break               % last chunk only testing
    end
    parameter.net.test_time(t) = toc(start_test);
    
    %% calculate correlation between hidden layers output and softmax output
    start_train = tic;
    if numHidLayer > 1
        parameter.HnCorrOut = {};
        for iHidLayer = 1:numHidLayer
            for iHiddenNodes = 2:parameter.ev{iHidLayer}.K+1
                HnCorrOut = zeros(1,M);
                for iIndexInp = 1:M
                    temporary    = corrcoef(parameter.net.activity{iHidLayer + 1}(:,iHiddenNodes),...
                        parameter.net.activityOutput(:,iIndexInp));
                    mm = size(temporary,1);
                    HnCorrOut(iIndexInp) = abs(temporary(1,mm));
                 
                end
                parameter.HnCorrOut{iHidLayer}(iHiddenNodes-1) = mean(HnCorrOut);
            end
            SCorr(iHidLayer) = mean(parameter.HnCorrOut{iHidLayer});
        end
        parameter.net.learningRate = zeros(1,numHidLayer);
        parameter.net.learningRate = 0.01*exp(-1*(1./SCorr-1));
    end
    
    %% Drift detection
    window = 1000;
    if mod(t,window) == 0 && t > 1000
        %% initiate drift detection parameter
        alpha_w = 0.005;
        alpha_d = 0.001;
        alpha   = 0.001;
        
        F_cut = parameter.squareError(window*countF_cut+...
            (1-window):window*countF_cut,:);
        if size(F_cut,2) > 1
            F_cut = mean(F_cut,2);
        end
        countF_cut = countF_cut + 1;
        cuttingpoint = 0;
        pp = size(F_cut,1);
        [b_A_Upper,~] = max(F_cut);
        [b_A_Lower,~] = min(F_cut);
        statisticA = mean(F_cut);
        for cut = 2:pp-1
            statisticB = mean(F_cut(1:cut,:));
            [a_A_Upper,~] = max(F_cut(1:cut,:));
            [a_A_Lower,~] = min(F_cut(1:cut,:));
            epsilon_B = (a_A_Upper - a_A_Lower)*sqrt(((cut)/(2*cut*(pp))*log(1/alpha)));
            epsilon_A = (b_A_Upper - b_A_Lower)*sqrt(((pp)/(2*cut*(pp))*log(1/alpha)));
            if (epsilon_A + statisticA)<=(epsilon_B + statisticB)
                cuttingpoint = cut;
                statistic_H = mean(F_cut(cuttingpoint+1:end,:));
                epsilon_D = (b_A_Upper-b_A_Lower)*sqrt(((pp - cuttingpoint)/(2*cuttingpoint*(pp - cuttingpoint)))*log(1/alpha_d));
                epsilon_W = (b_A_Upper-b_A_Lower)*sqrt(((pp - cuttingpoint)/(2*cuttingpoint*(pp - cuttingpoint)))*log(1/alpha_w));
                break
            end
        end
        if cuttingpoint == 0
            statistic_H = statisticA;
            epsilon_D = (b_A_Upper-b_A_Lower)*sqrt(((pp-cut)/(2*cut*(pp))*log(1/alpha_d)));
            epsilon_W = (b_A_Upper-b_A_Lower)*sqrt(((pp-cut)/(2*cut*(pp))*log(1/alpha_w)));
        end
        
        if abs(statisticB - statistic_H) > epsilon_D && st ~= 1 && cuttingpoint~=0
            st = 1;
            disp('Drift state: DRIFT');
            numHidLayer = numHidLayer + 1;
            parameter.net.hl = numHidLayer;
            parameter.net.nLayer = parameter.net.nLayer + 1;
            fprintf('The new Layer no %d is FORMED around chunk %d\n', numHidLayer, t)
            
            %% initiate NN weight parameters
            [ii,~] = size(parameter.net.weight{numHidLayer-1});
            parameter.net.weight{numHidLayer} = normrnd(0,sqrt(2/(ii+1)),[1,ii+1]);
            parameter.net.velocity{numHidLayer} = zeros(1,ii+1);
            parameter.net.grad{numHidLayer} = zeros(1,ii+1);
            
            %% initiate new classifier weight
            parameter.net.weightSoftmax = normrnd(0,1,[M,2]);
            parameter.net.velocitySoftMax = zeros(M,2);
            parameter.net.gradSoftMax = zeros(M,2);
            
            %% initiate iterative parameters
            parameter.ev{numHidLayer}.layer       = numHidLayer;
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
            parameter.ev{numHidLayer}.grow        = 0;
            parameter.ev{numHidLayer}.prune       = 0;
            
            %% check buffer
            if isempty(buffer_x)
                h = x;
                z = T;
            else
                h = [buffer_x;x];
                z = [buffer_T;T];
            end
            
            %% Constructing the input for next training
            h                           = [parameter.anomaly.Anomaliesx;h];
            h                           = [ones(size(h,1),1) h];
            parameter.net.activity{1}   = h;
            T                           = [parameter.anomaly.AnomaliesT;z];
            buffer_x                    = [];
            buffer_T                    = [];
            
            %% reset anomaly
            parameter.anomaly.indexAnomaly = 0;
            parameter.anomaly.currentInvCovA = 1*eye(I);
            parameter.anomaly.currentMeanC = zeros(1,I);
            parameter.anomaly.CACounter = 0;
            parameter.anomaly.ChangePoints = [];        % Index of identified change points
        elseif abs(statisticB - statistic_H) >= epsilon_W && abs(statisticB - statistic_H) < epsilon_D && st ~= 2
            st = 2;
            disp('Drift state: WARNING');
            buffer_x = x;
            buffer_T = T;
        else
            st = 3;
            disp('Drift state: STABLE');
            
            %% check buffer
            if isempty(buffer_x)
                
            else
                h                           = [buffer_x;x];
                h                           = [ones(size(h,1),1) h];
                parameter.net.activity{1}   = h;
                T                           = [buffer_T;T];
            end
            buffer_x                        = [];
            buffer_T                        = [];
        end
    else
        st                          = 3;
        disp('Drift state: STABLE');
        buffer_x                    = [];
        buffer_T                    = [];
    end
    drift(t)                        = st;
    HL(t)                           = numHidLayer;
    
    %% Discrinimanive training for all layers
    if st ~= 2
        disp('Discriminative Training: running ...');
        parameter = training(parameter,T,x_ori,T_ori);
        disp('Discriminative Training: ... finished');
    end
    parameter.net.update_time(t) = toc(start_train);
    
    %% clear current chunk data
    clear Data{t}
    parameter.net.activity = {};
    fprintf('=========Hidden layers were updated=========\n')
end
clc

%% save the numerical result
parameter.drift         = drift;
parameter.nFolds        = nFolds;
performance.update_time = [mean(parameter.net.update_time) std(parameter.net.update_time)];
performance.test_time   = [mean(parameter.net.test_time) std(parameter.net.test_time)];
performance.RMSE        = (mean(parameter.squareError)).^0.5;
performance.NDEI        = performance.RMSE./std(actualOutput);
performance.layer       = [mean(HL) std(HL)];
meanode                 = [];
stdnode                 = [];
for i = 1:parameter.net.hl
    a                       = nnz(~parameter.net.nodes{i});
    parameter.net.nodes{i}  = parameter.net.nodes{i}(a+1:t);
    meanode                 = [meanode mean(parameter.net.nodes{i})];
    stdnode                 = [stdnode std(parameter.net.nodes{i})];
end
performance.meanode             = meanode;
performance.stdnode             = stdnode;
performance.NumberOfParameters  = parameter.net.mnop;
parameter.HL                    = HL;


plot(predictedOutput)
ylim([0 1.1]);
xlim([1 nFolds]);
ylabel('i-th data point')
hold on
plot(actualOutput)
legend('predictedOutput','actualOutput')
end

% This code aims to construct neural network with several hidden layer
% one can choose to either connect every hidden layer output to
% the last output or not

function net = netInit(layer)
net.initialConfig        = layer;                               % Initial network configuration
net.nLayer               = numel(net.initialConfig);            %  Number of layer
net.hl                   = net.nLayer - 2;                      %  Number of hidden layer
net.activation_function  = 'sigmf';                              %  Activation functions of hidden layers: 'sigmf', 'tanh' and 'relu'
net.learningRate         = 0.01;                                %  Learning rate Note: typically needs to be lower when using 'sigmf' activation function and non-normalized inputs.
net.velocityCoeff        = 0.95;                                %  Momentum coefficient, higher value is preferred
net.output               = 'softmax';                            %  output layer can be selected as follows: 'sigmf', 'softmax', and 'linear'

%% initiate weights and weight momentum for hidden layer
for i = 2 : net.nLayer - 1
    net.weight {i - 1}      = normrnd(0,sqrt(2/(net.initialConfig(i-1)+1)),[net.initialConfig(i),net.initialConfig(i - 1)+1]);
    net.velocity{i - 1}     = zeros(size(net.weight{i - 1}));
    net.grad{i - 1}         = zeros(size(net.weight{i - 1}));
    net.c{i - 1}            = rand(net.initialConfig(i - 1),1);
end

%% initiate weights and weight momentum for output layer
net.weightSoftmax       = normrnd(0,sqrt(2/(size(net.weight {i - 1},1)+1)),[net.initialConfig(end),net.initialConfig(end - 1)+1]);
net.velocitySoftMax     = zeros(size(net.weightSoftmax));
net.gradSoftMax         = zeros(size(net.weightSoftmax));
end

function net = netInitTrain(layer)
net.initialConfig                       = layer;
net.nLayer                              = numel(net.initialConfig);
net.activation_function                 = 'sigmf';              %  Activation functions of hidden layers: 'sigmf' (sigmoid) or 'tanh_opt' (optimal tanh).
net.learningRate                        = 0.01;                 %  learning rate Note: typically needs to be lower when using 'sigmf' activation function and non-normalized inputs.
net.velocityCoeff                       = 0.95;                 %  Momentum
net.output                              = 'softmax';            %  output unit 'sigmf' (=logistic), 'softmax' and 'linear'
end

function net = testing(net, input, truevalue, ev)
%% feedforward
net = netFeedForward(net, input, truevalue);

%% calculate classification rate
net.index = net.hl;
net.squareError = (truevalue - net.activityOutput).^2;

for i = 1 : net.hl
    %% calculate the number of parameter
    if i == i
        [c,d] = size(net.weightSoftmax);
    else
        c = 0;
        d = 0;
    end
    [a,b] = size(net.weight{i});
    nop(i) = a*b + c*d;
    
    %% calculate the number of node in each hidden layer
    net.nodes{i}(net.t) = ev{i}.K;
end
net.nop(net.t) = sum(nop);
net.mnop = [mean(net.nop) std(net.nop)];
end

function net = netFeedForward(net, input, y)
nLayer = net.nLayer;
nData = size(input,1);
input = [ones(nData,1) input];      % by adding 1 to the first coulomn, it means the first coulomn of W is bias
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
        net.activityOutput = net.activity{nLayer - 1} * net.weightSoftmax';
        net.activityOutput = exp(net.activityOutput - max(net.activityOutput,[],2));
        net.activityOutput = net.activityOutput./sum(net.activityOutput, 2);
end

%% calculate error
net.error = y - net.activityOutput;

%% calculate loss function
switch net.output
    case {'sigmf', 'linear'}
        net.L = 1/2 * sum(sum(net.error .^ 2)) / nData;
    case 'softmax'
        net.L = -sum(sum(y .* log(net.activityOutput))) / nData;
end
end

function net = lossBackward(net)
nLayer = net.nLayer;
switch net.output
    case 'sigmf'
        backPropSignal{nLayer} = - net.error .* (net.activity{nLayer} .* (1 - net.activity{nLayer}));
    case {'softmax','linear'}
        backPropSignal{nLayer} = - net.error;          % dL/dy
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

function net = netFeedForwardTrain(net, x, y)

nLayer = net.nLayer;
nData = size(x,1);
net.activity{1} = x;

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
net.error = y - net.activity{nLayer};
end

function net = optimizerStep(net)
lr = [net.learningRate 0.01];
if numel(lr) ~= net.nLayer - 1
    lr = [lr 0.01];
end
for i = 1 : (net.nLayer - 1)
    if lr(i) > 0
        dW = net.grad{i};
        dW = lr(i) * dW;
        if(net.velocityCoeff > 0)
            net.velocity{i} = net.velocityCoeff*net.velocity{i} + dW;
            dW = net.velocity{i};
        end
        %% apply gradient
        net.weight{i} = net.weight{i} - dW;
    end
end
end

% MIT License
%
% Copyright (c) 2018 Andri Ashfahani Mahardhika Pratama
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
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

function parameter  = training(parameter,y,x_ori,T_ori)
%% initiate performance matrix
ka = 0;
indexAnomaly = parameter.anomaly.indexAnomaly;
indexStableExecution = size(parameter.net.activity{1},2);

%% initiate performance matrix
indexHL          = parameter.net.hl;
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
grow        = parameter.ev{indexHL}.grow;
prune       = parameter.ev{indexHL}.prune;

%% initiate training model
net = netInitTrain([1 1 1]);
net.activation_function = parameter.net.activation_function;
net.output = parameter.net.output;
net.nLayer = parameter.net.nLayer;
net.learningRate = parameter.net.learningRate;

%% substitute the weight to be trained to training model
for lyr = 1:parameter.net.nLayer - 1
    if lyr + 1 == parameter.net.nLayer
        net.weight{lyr}  = parameter.net.weightSoftmax;
        net.velocity{lyr} = parameter.net.velocitySoftMax;
        net.grad{lyr} = parameter.net.gradSoftMax;
    else
        net.weight{lyr}  = parameter.net.weight{lyr};
        net.velocity{lyr} = parameter.net.velocity{lyr};
        net.grad{lyr} = parameter.net.grad{lyr};
    end
end
[~,bb] = size(parameter.net.weight{indexHL});

%% load the data for training
x     = parameter.net.activity{1};
[N,I] = size(x);
kk    = randperm(N);
x     = x(kk,:);
y     = y(kk,:);

%% xavier initialization
if indexHL > 1
    n_in = parameter.ev{indexHL-1}.K;
else
    n_in = parameter.net.initialConfig(1);
end

%% main loop, train the model
for k = 1 : N
    kp = kp + 1;
    kl = kl + 1;
    indexAnomaly = indexAnomaly + 1;
    
    %% Incremental calculation of x_tail mean and variance
    [miu_x,std_x,var_x] = meanstditer(miu_x_old,var_x_old,parameter.net.activity{1}(k,:),kp);
    miu_x_old = miu_x;
    var_x_old = var_x;
    
    %% Expectation of z
    py = probit(miu_x,std_x)';
    for ii = 1:parameter.net.hl
        py = sigmf(net.weight{ii}*py,[1,0]);
        py = [1;py];
        if ii == 1
            Ey2 = py.^2;
        end
    end
    Ey = py;
    Ez = net.weight{lyr}*Ey;
    if parameter.net.hl > 1
        py = Ey2;
        for iiHiddenLayer = 2:parameter.net.hl
            py = sigmf(net.weight{iiHiddenLayer}*py,[1,0]);
            py = [1;py];
        end
        Ey2 = py;
    end
    Ez2 = net.weight{lyr}*Ey2;
    
    %% Network mean calculation
    bias2 = (Ez - y(k,:)').^2;
    ns = bias2;
    NS = norm(ns,'fro');
    
    %% Incremental calculation of NS mean and variance
    [miu_NS,std_NS,var_NS] = meanstditer(miu_NS_old,var_NS_old,NS,kl);
    miu_NS_old = miu_NS;
    var_NS_old = var_NS;
    miustd_NS = miu_NS + std_NS;
    
    if kl <= 2 || grow == 1
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
    if miustd_NS >= miustdmin_NS && kl > 2
        grow = 1;
        K = K + 1;
        fprintf('The new node no %d is FORMED around sample %d\n', K, k)
        node(k+1) = K;
        net.weight{indexHL} = [net.weight{indexHL};normrnd(0,sqrt(2/(n_in+1)),[1,bb])];
        net.velocity{indexHL} = [net.velocity{indexHL};zeros(1,bb)];
        net.grad{indexHL} = [net.grad{indexHL};zeros(1,bb)];
        net.weight{indexHL+1} = [net.weight{indexHL+1} normrnd(0,sqrt(2/(K+1)),[parameter.net.initialConfig(end),1])];
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
    [miu_NHS,std_NHS,var_NHS] = meanstditer(miu_NHS_old,var_NHS_old,NHS,kl);
    miu_NHS_old = miu_NHS;
    var_NHS_old = var_NHS;
    miustd_NHS = miu_NHS + std_NHS;
    
    if kl <= I+1 || prune == 1
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
    miustdmin_NHS = miumin_NHS + 2*(1.25*exp(-NHS)+0.75)*stdmin_NHS;
    VAR(kl,:)     = miu_NHS;
    
    %% pruning hidden unit
    if grow == 0 && K > 1 && miustd_NHS >= miustdmin_NHS && kl > I+1
        HS = Ey(2:end);
        [~,BB] = min(HS);
        fprintf('The node no %d is PRUNED around sample %d\n', BB, k)
        prune = 1;
        K = K - 1;
        node(k+1) = K;
        net.weight{indexHL}(BB,:) = [];
        net.velocity{indexHL}(BB,:) = [];
        net.grad{indexHL}(BB,:) = [];
        net.weight{indexHL+1}(:,BB+1) = [];
        net.velocity{indexHL+1}(:,BB+1) = [];
        net.grad{indexHL+1}(:,BB+1) = [];
    else
        node(k+1) = K;
        prune = 0;
    end
    
    %% feedforward
    net = netFeedForwardTrain(net, x(k,:), y(k,:));
    
    %% feedforward #2, executed if there is a hidden node changing
    net = lossBackward(net);
    net = optimizerStep(net);
    
    %% anomaly calculation
    if k <= size(x_ori,1)
        if indexAnomaly <= indexStableExecution
            parameter.anomaly.currentMeanC = meaniter(parameter.anomaly.currentMeanC,x_ori(k,:),indexAnomaly);
        elseif indexAnomaly > indexStableExecution
            mahaldist = (x_ori(k,:) - parameter.anomaly.currentMeanC)*parameter.anomaly.currentInvCovA*(x_ori(k,:) - parameter.anomaly.currentMeanC)';
            confCandidate = sort(net.activity{net.nLayer},'descend');
            y1 = confCandidate(1);
            y2 = confCandidate(2);
            confFinal = y1/(y1+y2);
            if (indexAnomaly > parameter.anomaly.StabilizationPeriod)
                
                if (mahaldist > parameter.anomaly.Threshold1 && mahaldist < parameter.anomaly.Threshold2) || confFinal <= 0.55 %\
                    ka = ka + 1;
                    indexAnomaly(ka) = k;
                    parameter.anomaly.CACounter = 0;
                else
                    parameter.anomaly.CACounter = parameter.anomaly.CACounter + 1;
                end
            end
            if(parameter.anomaly.CACounter >= parameter.anomaly.na)
                parameter.anomaly.ChangePoints = [parameter.anomaly.ChangePoints;indexAnomaly - parameter.anomaly.CACounter];
                parameter.anomaly.CACounter = 0;
            end
            [parameter.anomaly.currentInvCovA,parameter.anomaly.currentMeanC] = FormulatUpdate(parameter.anomaly.currentInvCovA,parameter.anomaly.currentMeanC,indexAnomaly,parameter.anomaly.Lambda,x_ori(k,:));
        end
    end
end

%% create buffer for anomaly
if ka ~= 0
    if size(parameter.anomaly.Anomaliesx,1) < 10000
        parameter.anomaly.Anomaliesx = [parameter.anomaly.Anomaliesx;parameter.net.activity{1}(indexAnomaly,2:end)];
        parameter.anomaly.AnomaliesT = [parameter.anomaly.AnomaliesT;T_ori(indexAnomaly,:)];
    elseif size(parameter.anomaly.Anomaliesx,1) >= 10000
        n_anomaly = size(parameter.net.activity{1}(indexAnomaly,2:end),1);
        parameter.anomaly.Anomaliesx = [parameter.anomaly.Anomaliesx(n_anomaly+1:end,:);parameter.net.activity{1}(indexAnomaly,2:end)];
        parameter.anomaly.AnomaliesT = [parameter.anomaly.AnomaliesT(n_anomaly+1:end,:);T_ori(indexAnomaly,:)];
    end
end

%% reset momentum and gradient
for lyr = 1:parameter.net.nLayer - 1
    if lyr + 1 == parameter.net.nLayer
        parameter.net.weightSoftmax = net.weight{lyr};
        parameter.net.velocitySoftMax = net.velocity{lyr}*0;
        parameter.net.gradSoftMax = net.grad{lyr}*0;
    else
        parameter.net.weight{lyr} = net.weight{lyr};
        parameter.net.velocity{lyr} = net.velocity{lyr}*0;
        parameter.net.grad{lyr} = net.grad{lyr}*0;
    end
end

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
parameter.ev{indexHL}.grow        = grow;
parameter.ev{indexHL}.prune       = prune;
end

function p = probit(miu,std)
p = (miu./(1 + pi.*(std.^2)./8).^0.5);
end

function [miu] = meaniter(miu_old,x,k)
miu = miu_old + (x - miu_old)./k;
end

function [miu,std,var] = meanstditer(miu_old,var_old,x,k)
miu = miu_old + (x - miu_old)./k;
var = var_old + (x - miu_old).*(x - miu);
std = sqrt(var/k);
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


%% stable softmax
function output = stableSoftmax(activation,weight)
output = activation * weight';
output = exp(output - max(output,[],2));
output = output./sum(output, 2);
end


