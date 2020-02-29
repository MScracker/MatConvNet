function patches = samplePatches(rawImages, patchSize, numPatches)
% rawImages is of size (image width)*(image height) by number_of_images
% We assume that image width = image height
d = size(rawImages);
assert(numel(d)>2,'rawImages dementions must be flatterned ');
imWidth = size(rawImages,1);
imHeight = size(rawImages,2);

if(numel(d) > 3)
	imChannels = size(rawImages,3) 
	numImages = size(rawImages,4);
else
	numImages = size(rawImages,3);
end
% Initialize patches with zeros.  
patches = zeros(patchSize*patchSize*imChannels, numPatches);

% Maximum possible start coordinate
maxWidth = imWidth - patchSize + 1;
maxHeight = imHeight - patchSize + 1;

% Sample!
for num = 1:numPatches
    x = randi(maxHeight);
    y = randi(maxWidth);
    img = randi(numImages);
	if (numel(d) > 3)
		p = rawImages(x:x+patchSize-1,y:y+patchSize-1,:,img);
	else
		p = rawImages(x:x+patchSize-1,y:y+patchSize-1,img);
	end
    patches(:,num) = p(:);
end
    

