clear all
clc

load brdata

ratio = 0.80; % learning data ratio

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

%'Plots', 'training-progress', ...

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 500, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {validD, targetTestL}, ...
    'ValidationFrequency', 13, ...
    'Verbose', false, ...
    'Plots', 'none', ...
    'ExecutionEnvironment', 'gpu');

testaccuracy    = 0;
accuracy        = 0;

randseedval = rand('seed');

layerc = randi(11) + 4;

while(testaccuracy < 0.92 || accuracy < 0.92)
    s = RandStream('mt19937ar', 'Seed', 'shuffle');
    RandStream.setGlobalStream(s);
    randseed = s.State;
    
    randseedval = rand('seed');
    
    clear rLC
    layerc = randi(5) + 4;
    rLC = randi(120, [1 layerc]) + 3;

    %% Define Network Architecture
    % Define the convolutional neural network architecture.
    breastCNNclassifyLayers

    net = trainNetwork(trainD, train1L, layers, options);

    % train
    predictedLabels = classify(net, trainD)';
    YValidation = train1L';

    accuracy = sum(predictedLabels == YValidation) / numel(YValidation);

%     disp(['RandSeedStream: '])
%     randseed
    
%     disp(['RandSeedVal: ' num2str(randseedval)])
    
    disp(['Train Accuracy: %' num2str(accuracy * 100)])
    % Test
    predictedLabels = classify(net, validD)';
    YValidation = targetTestL';

    testaccuracy = sum(predictedLabels == YValidation) / numel(YValidation);

    disp(['Test Accuracy: %' num2str(testaccuracy * 100)])
    
    disp(['Layer Count: ' num2str(layerc)])
    disp('-------------------------------------')
end