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

sel_unlabel = randperm(size(X_unlabeled,4));
numPatches = 200000;
patches = samplePatches(X_unlabeled(:,:,:,sel_unlabel(1:20000)),8,8,numPatches);

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
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  patches = reshape(z, 8, 8, 3, []) ;
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