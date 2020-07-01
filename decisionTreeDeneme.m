clear;
x=xlsread('degerler.xlsx');
y= xlsread('turler.xlsx');
% c = cvpartition(y,'KFold',3);
% fun = @(xTrain,yTrain,xTest,yTest)(sum(~strcmp(yTest,...
%     classify(xTest,xTrain,yTrain)))); 
% rate = sum(crossval(fun,x,y,'partition',c))...
%            /sum(c.TestSize)
rng(1); % For reproducibility
MdlDefault = fitctree(x,y,'CrossVal','on');
numBranches = @(x)sum(x.IsBranch);
mdlDefaultNumSplits = cellfun(numBranches, MdlDefault.Trained);

figure;
histogram(mdlDefaultNumSplits)
view(MdlDefault.Trained{1},'Mode','graph')
classErrorDefault = kfoldLoss(MdlDefault)
Mdl = fitctree(x,y,'OptimizeHyperparameters','auto')

predictLabels = predict(Mdl,x);            %Evaluate on test dataset 
trueLabels = y(:,1); 
testAccuracy = sum(predictLabels == trueLabels)/length(trueLabels);

