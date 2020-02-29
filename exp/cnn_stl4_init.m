function net = cnn_stl4_init(varargin)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST
opts.useBnorm = true ;
opts.pretraining = true ;
opts = vl_argparse(opts, varargin) ;

imageChannels = 3;     % number of channels (rgb, so 3)
patchDim = 8;          % patch dimension
visibleSize = patchDim * patchDim * imageChannels;  % number of input units 
hiddenSize = 400;
load data/stl4/STL10Features.mat
W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
W_white = W*ZCAWhite;
b_white = b-W*ZCAWhite*meanPatch;
filters = single(reshape(W_white',8,8,3,400));
biases = single(b_white');
rng('default');
rng(0) ;
f = 1/100;
net.layers = {} ;
if opts.pretraining
	net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{filters, biases}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
else 
	net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(8,8,3,400,'single'),zeros(1,400,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
end						   
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [19 19], ...
                           'stride', 19, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(3,3,400,4,'single'),zeros(1,4,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;						   
net.layers{end+1} = struct('type', 'sigmoid') ;	
net.layers{end+1} = struct('type', 'softmaxloss') ;

% optionally switch to batch normalization
if opts.useBnorm
  net = insertBnorm(net, 1) ;
  net = insertBnorm(net, 4) ;
  net = insertBnorm(net, 7) ;
end

% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1], ...
               'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
