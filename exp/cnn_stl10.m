function [ net info ] = cnn_stl10( varargin )
%CNN_STL10 Summary of this function goes here
%   Detailed explanation goes here
run(fullfile(fileparts(mfilename('fullpath')),'..', 'setup.m')) ;

opts.expDir = fullfile('data','stl10') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data','stl10') ;
opts.imdbPath = fullfile(opts.expDir, 'stl10_imdb.mat');
opts.useBnorm = false ;
opts.whitenData = true;
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

%[net, info] = cnn_train(net, imdb, @getBatch, opts.train, 'val', find(imdb.images.set == 3)) ;

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
set = [ones(1,numel(y_train)) 2*ones(1,numel(size(X_unlabeled,4)) 3*ones(1,numel(y_test))];
data = single(cat(4, X_train, X_test));
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;

%sampling patches 
numPatches = 500000;
patchDim  = 8;          % patch dimension
colorChannels = 3;
sel_unlabel = randperm(size(X_unlabeled,4));
patches = samplePatches(X_unlabeled(:,:,:,sel_unlabel(1:50000)),patchDim,patchDim,numPatches);

%remove mean of patches
patchMean = mean(patches, 2);  
patches = bsxfun(@minus, patches, patchMean);
patches = single(patches);

if opts.whitenData
  z = reshape(patches,[],numPatches) ;
  W = z*z'/numPatches ;
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  ZCAWhite = V*diag(en./max(sqrt(d2), 10))*V';
  imdb.images.ZCAWhite = ZCAWhite;
  z = ZCAWhite*z;
  patches = reshape(z, patchDim, patchDim, colorChannels, []) ;
end

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.patches = patches;
imdb.images.patchMean = patchMean;
trainLabels = y_train';
testLabels = y_test';
imdb.images.labels = cat(2, trainLabels, testLabels) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'unlabeled', 'test'} ;
imdb.meta.classnames = class_names;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:10,'uniformoutput',false) ;
end