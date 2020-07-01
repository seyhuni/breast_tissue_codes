clc;
close all;
clear all;

%% Training Side
x=xlsread('degerler.xlsx');
y= xlsread('turler.xlsx');
data=x;
target=y;
% create a decision tree
dtree = fitctree(data,target,'MinParentSize',2); % dtree is the trained model. save it at end for doing testing


% train performance
label = predict(dtree,data);

perf=sum(label==target)/size(label,1) % performance in the range of 0 to 1

%% Testing Side
% for testing load the trained model
load('dtree.mat');
testdata = [x(1:23,:);x(44:55,:)]; % take 1 new unknown observation and give to trained model
Group = predict([dtree.Y(1:23,:);dtree.Y(44:55,:)],testdata);

testAccuracy = sum(Group == [y(1:23,:);y(44:55,:)])/length(Group);

