function [ net info ] = cnn_stl10( varargin )
%CNN_STL10 Summary of this function goes here
%   Detailed explanation goes here

opts.expDir = fullfile('data','stl10') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data','stl10') ;
opts.imdbPath = fullfile(opts.dataDir, 'stl10_imdb.mat');
opts.useBnorm = false ;
opts.whitenData = true;
opts.contrastNormalization = true ;
opts.train.batchSize = 100 ;
opts.train.numEpochs = 20;
opts.train.continue = true ;
opts.train.gpus = [] ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;
opts.train.plotDiagnostics = false;
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getSTLImdb(opts) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

net = cnn_stl10_init(opts) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

%[net, info] = cnn_train(net, imdb, @getBatch, opts.train) ;

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

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
set = [ones(1,numel(y_train))  3*ones(1,numel(y_test))];
data = single(cat(4, X_train, X_test));
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;


if opts.whitenData
  z = reshape(data,[],13000) ;
  sigma = z(:,set == 1)*z(:,set == 1)'/5000 ;
  [U,S,V] = svd(sigma);
  z = U * diag(1./sqrt(diag(S))) * U' * z;
  dataWhite = reshape(z, 96, 96, 3, []) ;
  imdb.images.data_white = dataWhite;
end

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
trainLabels = y_train';
testLabels = y_test';
imdb.images.labels = cat(2, trainLabels, testLabels) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classnames = class_names;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:10,'uniformoutput',false) ;