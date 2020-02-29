run(fullfile(fileparts(mfilename('fullpath')),'..', 'setup.m')) ;

% --------------------------------------------------------------------
%                    Apply preprocessing
% --------------------------------------------------------------------
%initializing 
load stlSampledPatches.mat;
epsilon = 0.1;	       % epsilon for ZCA whitening

% Subtract mean patch (hence zeroing the mean of the patches)
numPatches = size(patches,2);
meanPatch = mean(patches, 2);  
patches = bsxfun(@minus, patches, meanPatch);
% Apply ZCA whitening
sigma = patches * patches' / numPatches;
[u, s, v] = svd(sigma);
ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';
patches = ZCAWhite * patches;

data = patches';
% --------------------------------------------------------------------
%                    Learning feature
% --------------------------------------------------------------------
%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
rand('state',0)
sae = saesetup([192 400]);
sae.ae{1}.activation_function              = 'sigm';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
sae.ae{1}.learningRate                     = 0.01;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
sae.ae{1}.momentum                         = 0.5;          %  Momentum
sae.ae{1}.scaling_learningRate             = 1;            %  Scaling factor for the learning rate (each epoch),value '1' means unchange learning rate. 
sae.ae{1}.weightPenaltyL2                  = 3e-3;            %  L2 regularization
sae.ae{1}.nonSparsityPenalty               = 5;            %  Non sparsity penalty
sae.ae{1}.sparsityTarget                   = 0.035;         %  Sparsity target
sae.ae{1}.inputZeroMaskedFraction          = 0;            %  Used for Denoising AutoEncoders
sae.ae{1}.dropoutFraction                  = 0;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
sae.ae{1}.testing                          = 0;            %  Internal variable. nntest sets this to one.
sae.ae{1}.output                           = 'linear';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
for i = 1:numel(sae.ae)
	fprintf('sae.ae{%d}:',i);
	sae.ae{i}
	fprintf('\n');
end
opts.numepochs =  400;
opts.batchsize = 500;
sae = saetrain(sae, data, opts);
displayColorNetwork(sae.ae{1}.W{1}(:,2:end)');
W = sae.ae{1}.W{1}(:,2:end);
b = sae.ae{1}.W{2}(:,2:end);


