function [net, info] = cnn_mnist(varargin)
% CNN_MNIST  Demonstrated MatConNet on MNIST

run(fullfile(fileparts(mfilename('fullpath')),...
  '..','matlab', 'vl_setupnn.m')) ;

opts.pretraining = false;  
  
if opts.pretraining 
	opts.expDir = fullfile('data','stl4') ;
else
	opts.expDir = fullfile('data','stl4_pretraining') ;
end
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data','stl4') ;
opts.imdbPath = fullfile(opts.dataDir, 'stl_imdb.mat');
opts.useBnorm = false ;
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

net = cnn_stl4_init('useBnorm', opts.useBnorm,'pretraining',opts.pretraining) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

[net, info] = cnn_train(net, imdb, @getBatch, opts.train, ...
    'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function imdb = getSTLImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
load(fullfile(opts.expDir, 'stlTrainSubset.mat'));
load(fullfile(opts.expDir, 'stlTestSubset.mat'));

set = [ones(1,numel(trainLabels)) 3*ones(1,numel(testLabels))];
data = single(cat(4, trainImages, testImages));
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
trainLabels = trainLabels';
testLabels = testLabels';
imdb.images.labels = cat(2, trainLabels, testLabels) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:4,'uniformoutput',false) ;
