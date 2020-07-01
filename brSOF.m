%% Self-Organising Fuzzy Logic (SOF) classifier for Breast Cancer Dataset
clear all
clc
close all

load brdata

%% The SOF classifier conducts offline learning from static datasets
Input.TrainingData      = cancerInputs';                % Input data samples
Input.TrainingLabel     = cancerTargetsSingleRow';      % Labels of the input data samples
GranLevel               = 12;                           % Level of granularity (Once being fixed in offline training stage, it cannot be changed further)
DistanceType            = 'Mahalanobis';                % Type of distance/dissimilarity SOF classifier uses, which can be 'Mahalanobis', 'Cosine' or 'Euclidean'
Mode                    = 'OfflineTraining';            % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output1]               = SOFClassifier(Input, GranLevel, Mode, DistanceType); 
% Output1.TrainedClassifier  - Offline primed SOF classifier

%% The SOF classifier conducts validation on testing data
Input                   = Output1;                      % Offline primed SOF classifier
Input.TestingData       = cancerInputs';                % Testing 
Input.TestingLabel      = cancerTargetsSingleRow';      % Labels of the tetsing data samples
Mode                    = 'Validation';                 % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]               = SOFClassifier(Input, GranLevel, Mode, DistanceType);
% Output2.TrainedClassifier  - Trained SOF classifier (same as the input)
% Output2.EstimatedLabel      - Estimated label of validation data
% Output2.ConfusionMatrix     - confusion matrix of the result

Output2.ConfusionMatrix

testAccuracy = sum(Output2.EstimatedLabel==cancerTargetsSingleRow')/length(cancerTargetsSingleRow);
