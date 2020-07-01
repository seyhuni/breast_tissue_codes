clear layers

layers = [
        imageInputLayer([9 1 1]) % 9x1x1 refers to number of features per sample

        convolution2dLayer(3, rLC(1), 'Padding','same')
        batchNormalizationLayer
        reluLayer];
   
for p = 2 : layerc
    layers = [
        layers
        
        convolution2dLayer(3, rLC(p), 'Padding','same')
        reluLayer];
end
    
layers = [
        layers
        
        fullyConnectedLayer(6)
        softmaxLayer
        classificationLayer];