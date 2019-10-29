% MIT License
% 
% Copyright (c) 2019  Choiru Za'in Andri Ashfahani Mahardhika Pratama
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
clear
clc

%% Load the Dataset (This is for running all datasets)
for j=6 % modify the index if you want to specify datasets (for example: j=4:4), means that only running RFID dataset
    if (j==1)
        load 01SUSY;        nInput = 18;     name='SUSY';
    elseif (j==2)
        load 02HEPMASS19;     nInput = 28;     name='HEPMASS';
    elseif (j==3)
        load 03RLCPS;       nInput = 9;      name='RLCPS';
    elseif (j==4)
        load 04RFID;        nInput = 3;      name='RFID';
    elseif (j==5)
        load 05Mnist;       nInput = 784;    name='Mnist';
    elseif (j==6)
        load 06RMnist;      nInput = 784;    name='RMnist';
    elseif (j==7)
        load 07PMnist;      nInput = 784;    name='PMnist';                 
    elseif (j==8)
        load 08KDD;         nInput = 41;     name = 'KDD';
    elseif (j==9)
        load 09SEA;         nInput = 3;      name='SEA';
    else
        load 10HYPERPLANE;  nInput = 4;      name='HYPERPLANE';
    end
    
end

%% RUN NADINE for Five times experiment
for i=1:1 % modify the index if we want to run several times of experiment (for example: i=1:10), means that we want to run 10 times experiment
    name2=strcat(num2str(i),name);
    [parameter,performance] = NADINE_classification(data,nInput,name);
end




