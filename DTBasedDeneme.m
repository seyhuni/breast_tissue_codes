clc;
close all;
clear all;

%% Training Side
x=xlsread('degerler.xlsx');
y= xlsread('turler.xlsx');

trainingData=[x(1:5,:);x(15:28,:);x(35:55,:);x(60:74,:);x(80:96,:);x(99:103,:)];
trainingLabel=[y(1:5,:);y(15:28,:);y(35:55,:);y(60:74,:);y(80:96,:);y(99:103,:)];

dtree = fitctree(trainingData,trainingLabel,'MinParentSize',2); % dtree is the trained model. save it at end for doing testing

label = predict(dtree,trainingData);

perf=sum(label==trainingLabel)/size(label,1) % performance in the range of 0 to 1

%% Test Side

testData = [x(6:14,:);x(29:34,:);x(56:59,:);x(75:79,:);x(97:98,:);x(104:106,:)]; % take 1 new unknown observation and give to trained model
Group = predict(dtree,testData);

testAccuracy = sum(Group == [y(6:14,:);y(29:34,:);y(56:59,:);y(75:79,:);y(97:98,:);y(104:106,:)])/length(Group);

labels = [y(6:14,:);y(29:34,:);y(56:59,:);y(75:79,:);y(97:98,:);y(104:106,:)];
scores = [x(6:14,1);x(29:34,1);x(56:59,1);x(75:79,1);x(97:98,1);x(104:106,1)];
[fpr,tpr] = perfcurve(labels,Group,1)
