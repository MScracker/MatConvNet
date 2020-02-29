
%function test_example_SAE
train_x = double(train_x)/255;
test_x  = double(test_x)/255;
train_y = double(train_y);
test_y  = double(test_y);

%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
rand('state',0)
sae = saesetup([784 200 100]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;
opts.numepochs =   1;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);
visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([784 200 100 10]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
nn.W{1} = sae.ae{1}.W{1};

% Train the FFNN
opts.numepochs =   1;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.16, 'Too big error');



% --------------------------------------------------------------------
function imdb = getSTLImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
load(fullfile(opts.dataDir, 'train.mat'));
X_train = reshape(X',96,96,3,5000);
y_train = y;
load(fullfile(opts.dataDir, 'test.mat'));
X_test = reshape(X',96,96,3,8000);
y_test = y;
load(fullfile(opts.dataDir, 'unlabeled.mat'));
X_unlabeled = reshape(X',96,96,3,100000);
set = [ones(1,numel(y_train)) 2*ones(1,numel(size(X_unlabeled,4)) 3*ones(1,numel(y_test))];

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
trainLabels = y_train';
testLabels = y_test';
imdb.images.labels = cat(2, trainLabels, testLabels) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'unlabeled', 'test'} ;
imdb.meta.classnames = class_names;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:10,'uniformoutput',false) ;
