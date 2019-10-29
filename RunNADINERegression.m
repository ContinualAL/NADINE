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
% SOFTWARE.

clc
clear
close all

%% load data
load('SP500.mat'); nInput = 5;
% load('HouseholdElectricPower.mat'); nInput =4;
% load('ConditionMonitoring.mat'); nInput = 16;

%% run stacked autonomous deep learning
for h = 1:5
    clc
    clearvars -except h data nInput result
    close all
    [parameter,performance] = NADINE_regression(data,nInput);
    result.error(h,:) = performance.RMSE;
    result.update_time(h) = sum(parameter.net.update_time);
    result.training_time(h) = sum(parameter.net.test_time);
    result.net{h} = parameter.net;
    result.performance{h} = performance;
end
clear data
disp(result)