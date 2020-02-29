% Configuration
imageDim = 28;
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
filterDim = 9;    % Filter size for conv layer
numFilters = 20;   % Number of filters for conv layer
poolDim = 2;      % Pooling dimension, (should divide imageDim-filterDim+1)


trainImages = loadMNISTImages('../data/train-images-idx3-ubyte');
trainLabels = loadMNISTLabels('../data/train-labels-idx1-ubyte');
trainLabels(trainLabels == 0) = 10;
testImages = loadMNISTImages('../data/t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('../data/t10k-labels-idx1-ubyte');
testLabels(testLabels == 0) = 10;

%train_x = double(reshape(train_x',28,28,60000))/255;
train_x = reshape(trainImages,imageDim,imageDim,[]);
%test_x = double(reshape(test_x',28,28,10000))/255;
test_x = reshape(testImages,imageDim,imageDim,[]);
train_y = full(sparse(trainLabels,1:length(trainLabels),1)); 
test_y = full(sparse(testLabels,1:length(testLabels),1));

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

rand('state',0)

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 20, 'kernelsize', 9) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
};


opts.alpha = 1e-1;
opts.batchsize = 250;
opts.numepochs = 20;

cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);
fprintf('error rate is %f\n',er);
%plot mean squared error
figure; plot(cnn.rL);
assert(er<0.12, 'Too big error');
