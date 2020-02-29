function res =testdisplay(imdb,net,pre,post)
m = randperm(numel(find(imdb.images.set==1)));
[im,labs] = getBatch(imdb,m(1:100));
res = [];
res = vl_simplenn(net,im,[],res);
figure ; colormap gray ;
vl_imarraysc(squeeze(res(pre).x(:,:,1,:)),'spacing',1)
figure ; colormap gray ;
vl_imarraysc(squeeze(res(post).x(:,:,1,:)),'spacing',1)
end