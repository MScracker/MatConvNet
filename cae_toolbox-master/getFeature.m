root = fileparts(mfilename('fullpath'));
X = load(fullfile(root,'..','data','stl10', 'unlabeled.mat'));
indx = randperm(size(X,1));
sel = 10000;

x_train = reshape(X(indx(1:sel))',96,96,3,[]);

% input channels | output channels | kernel size | pool size | noise
cae = cae_setup(3,64,5,2,0);
opts.alpha = 0.03;
opts.numepochs = 8;
opts.batchsize = 100;
opts.shuffle = 1;
opts.convapi = true;
cae = cae_train(cae, x, opts);
