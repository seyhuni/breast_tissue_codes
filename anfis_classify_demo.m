%% Initialize
clc
clear
close all
warning off all

%% Load Data
load brdata

ratio = 0.90; % learning data ratio

%load brdata
splitMyData

cancerInputs = train1D';
cancerTargets = train1L;
cancerTargetsSingleRow = train1L;
%% Classification  Demo 1
epoch_n = 50;
dispOpt = zeros(1,4);
numMFs = 10;
inmftype= 'gbellmf';
outmftype= 'linear';
split_range=2;
Model=ANFIS.train(train1D',train1L,split_range,numMFs,inmftype,outmftype,dispOpt,epoch_n);
disp('Model')
disp(Model)
Result=ANFIS.classify(Model,train1D');
%Performance Calculation
Rvalue=@(a,b)(1-abs((sum((b-a).^2)/sum(a.^2))));
RMSE=@(a,b)(abs(sqrt(sum((b-a).^2)/length(a))));
MAPE=@(a,b)(abs(sum(sqrt((b-a).^2)*100./a)/length(a)));
fprintf('Anfis     R      RMSE     MAPE\n')
r=Rvalue(train1L, Result);
rmse=RMSE(train1L, Result);
mape=MAPE(train1L, Result);
fprintf('Anfis  %.4f\t%.4f\t%.4f\n',r,rmse,mape);
%% Display
disp('TrainData Predicted ')
disp([train1L(1:end), Result(1:end)])

% test
Result2=ANFIS.classify(Model, trainTest');
fprintf('Anfis     R      RMSE     MAPE\n')
r=Rvalue(targetTestL, Result2);
rmse=RMSE(targetTestL, Result2);
mape=MAPE(targetTestL, Result2);
fprintf('Anfis  %.4f\t  %.4f\t  %.4f\n',r,rmse,mape);
%% Display
n=40;
disp('TestData Predicted ')
disp([targetTestL(1:end),Result2(1:end)])

%% Rounding Classification  Demo 2
% epoch_n = 30;
% dispOpt = zeros(1,4);
% numMFs = 3;
% inmftype= 'gbellmf';
% outmftype= 'linear';
% split_range=3;
% Model=ANFIS.train(cancerInputs,round(cancerTargetsSingleRow),split_range,numMFs,inmftype,outmftype,dispOpt,epoch_n);
% disp('Model')
% disp(Model)
% Result=round(ANFIS.classify(Model,cancerInputs));
% % Performance Calculation
% Accuracy=mean(round(cancerTargetsSingleRow)==Result);
% disp('Accuracy')
% disp(Accuracy)
% %% Display
% n=100;
% disp('TestClass Predicted ')
% disp(round([cancerTargetsSingleRow(1:n),Result(1:n)]))
