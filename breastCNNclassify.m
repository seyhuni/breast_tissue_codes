clear all
clc

load brdata

ratio = 0.75; % learning data ratio

%load brdata
splitMyData

[dX, dY] = size(train1D);
for t = 1 : dY
    trainD(:, 1, 1, t) = train1D(:, t);
end

[dX, dY] = size(trainTest);
for t = 1 : dY
    validD(:, 1, 1, t) = trainTest(:, t);
end

%% Define Network Architecture
% Define the convolutional neural network architecture.
layers = [
    imageInputLayer([9 1 1]) % 9x1x1 refers to number of features per sample
    
    convolution2dLayer(3, 4, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 8, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 16, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 32, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 64, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 32, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    %fullyConnectedLayer(318) % 384 refers to number of neurons in next FC hidden layer
    %fullyConnectedLayer(318) % 384 refers to number of neurons in next FC hidden layer
    fullyConnectedLayer(6) % 2 refers to number of neurons in next output layer (number of output classes)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 2000, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {validD, targetTestL}, ...
    'ValidationFrequency', 13, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'cpu');

net = trainNetwork(trainD, train1L, layers, options);

% train
predictedLabels = classify(net, trainD)';
YValidation = train1L';

accuracy = sum(predictedLabels == YValidation) / numel(YValidation);

disp(['Train Accuracy: %' num2str(accuracy * 100)])
% Test
predictedLabels = classify(net, validD)';
YValidation = targetTestL';

accuracy = sum(predictedLabels == YValidation) / numel(YValidation);

disp(['Test Accuracy: %' num2str(accuracy * 100)])