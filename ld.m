setup
imageDim = 64;         % image dimension
imageChannels = 3;     % number of channels (rgb, so 3)

patchDim = 8;          % patch dimension
numPatches = 50000;    % number of patches

visibleSize = patchDim * patchDim * imageChannels;  % number of input units 
outputSize = visibleSize;   % number of output units
hiddenSize = 400;
load data/stl4/STL10Features.mat
W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
W_white = W*ZCAWhite;
b_white = b-W*ZCAWhite*meanPatch;
filters = single(reshape(W_white',8,8,3,400));
biases = single(b_white');


opts.expDir = fullfile('data','stl4') ;
opts.dataDir = fullfile('data','stl4') ;
opts.imdbPath = fullfile(opts.expDir, 'stl_imdb.mat');
if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
	warning('data missing!');
end
batch = 1:100;
[im,labels] = getBatch(imdb, batch);
convout = vl_nnconv(im, filters, biases, ...
                               'pad', 0, 'stride', 1) ;
poolout = vl_nnpool(convout, 19, ...
                    'pad', 0, 'stride', 19, ...
                    'method', 'max') ;
out = vl_nnsoftmaxloss(poolout, labels) ;