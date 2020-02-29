function net = cnn_stl10_init(imdb,varargin)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST
opts.useBnorm = true ;
opts = vl_argparse(opts, varargin) ;
[W, b] = learnFeatures(imdb.images.patches);
W_white = W*imdb.images.ZCAWhite;
b_white = b-W*imdb.images.ZCAWhite*imdb.images.meanPatch;
filters = single(reshape(W_white',8,8,3,400));
biases = single(b_white');
rng('default');
rng(0) ;
f = 1/100;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{filters, biases}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'sigmoid') ;							   
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
