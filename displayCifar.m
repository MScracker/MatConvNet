function res =displayCifar(data,net,pre,post)
m = randperm(size(data,4));
im = data(:,:,:,m(1:100));
res = [];
res = vl_simplenn(net,im,[],res);
figure ; colormap gray ;
vl_imarraysc(squeeze(res(pre).x(:,:,1,:)),'spacing',1)
figure ; colormap gray ;
vl_imarraysc(squeeze(res(post).x(:,:,1,:)),'spacing',1)
end