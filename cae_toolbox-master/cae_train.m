% train the cae using backpropagation and stochastic gradient descent.
% the pooling operation will filter out all non-maximum values in the neighbourhood, 
% which will result in a sparse matrix.
% no subsampling is used according to the original paper Masci 2011.
% cae_check_grad can be turned on to verify the gradients numerically.

function [cae] = cae_train(cae, x, opts)
    
    [x,para] = cae_check(cae,x,opts);
    cae.L = zeros(opts.numepochs*para.bnum,1);
    for i=1:opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        tic;
        if opts.shuffle==1
            idx = randperm(para.pnum);
        else
            idx = linspace(1,para.pnum,para.pnum);
        end
        for j = 1 : para.bnum
            batch_x = x(:,:,:,idx((j-1)*para.bsze+1:j*para.bsze));            
            cae = cae_ffbp(cae, batch_x, para);
%             [numdw,numdb,numdc] = cae_check_grad(cae, batch_x, para); % correct for multi channel input data
            cae = cae_update(cae, opts); % w w_tilde            
            cae.L((i-1)*para.bnum+j)=cae.loss;            
        end
        disp(mean(cae.L((i-1)*para.bnum+1:i*para.bnum)));
        toc;
    end    
end

function [cae] = cae_ffbp(cae, x, para)
    x_noise = x.*(rand(size(x))>=cae.noise);
	if opts.convapi
		cae.h = vl_nnconv(x_noise, cae.w, cae.b, 'pad', [0 1 0 1], 'stride', 1) ;
		cae.h = sigm(cae.h + reshape(repmat(cae.b,1,para.bsze),1,1,size(cae.b,1),[]);
	else
		cae = cae_up(cae, x_noise, para);
    end
	cae = cae_pool(cae, para);
%     cae = cae_resize_pool(cae, para);
	if opts.convapi
		cae.h = vl_nnconv(cae.o, cae.w_tilde, cae.b, 'pad', [0 1 0 1], 'stride', 1) ;
	else
		cae = cae_down(cae, para);
    end
    cae = cae_grad(cae, x, para);
end

function [cae] = cae_up(cae, x, para)
    % ks: kernel size, oc: output channels
    cae.h = zeros(para.m-cae.ks+1,para.m-cae.ks+1,cae.oc,para.bsze);
    for pt = 1:para.bsze
        for oc = 1:cae.oc
            for ic = 1:cae.ic
                cae.h(:,:,oc,pt) = cae.h(:,:,oc,pt) + conv2(x(:,:,ic,pt),cae.w(:,:,ic,oc),'valid');
            end
            cae.h(:,:,oc,pt) = sigm(cae.h(:,:,oc,pt)+cae.b(oc));
        end        
    end
end

function cae = cae_pool(cae, para)
    % ps: pool size
    if cae.ps>=2
        cae.h_pool = zeros(size(cae.h));
        cae.h_mask = zeros(size(cae.h));
        for i = 1:para.pgrds
            for j = 1:para.pgrds
                grid = cae.h((i-1)*cae.ps+1:i*cae.ps,(j-1)*cae.ps+1:j*cae.ps,:,:);           
                mx = repmat(max(max(grid)),cae.ps,cae.ps);
                mask = (grid==mx);
                sparse_grid = zeros(size(grid));
                sparse_grid(mask) = grid(mask);
                cae.h_pool((i-1)*cae.ps+1:i*cae.ps,(j-1)*cae.ps+1:j*cae.ps,:,:) = sparse_grid;
                cae.h_mask((i-1)*cae.ps+1:i*cae.ps,(j-1)*cae.ps+1:j*cae.ps,:,:) = mask;
            end
        end        
    end
end

function [cae] = cae_down(cae, para)
    % ks: kernel size, oc: output channels
    cae.o = zeros(para.m,para.m,cae.ic,para.bsze);
    for pt = 1:para.bsze
        for ic = 1:cae.ic
            for oc = 1:cae.oc
                cae.o(:,:,ic,pt) = cae.o(:,:,ic,pt) + conv2(cae.h_pool(:,:,oc,pt),cae.w_tilde(:,:,ic,oc),'full');
            end
            cae.o(:,:,ic,pt) = sigm(cae.o(:,:,ic,pt)+cae.c(ic));
        end        
    end
end

function [cae] = cae_grad(cae, x, para)
    % o = sigmoid(y'), y' = sigma(maxpool(sigmoid(h'))*W~)+c, h' = W*x+b
    % y', h' are pre-activation terms
    cae.err = (cae.o-x);
    cae.loss = 1/2 * sum(cae.err(:) .^2 )/para.bsze;

    % dloss/dy' = (y-x)(y(1-y))
    cae.dy = cae.err.*(cae.o.*(1-cae.o))/para.bsze;
    % dloss/dc = sigma(dy')
    cae.dc = reshape(sum(sum(cae.dy)),[size(cae.c) para.bsze]);
    % dloss/dmaxpool(sigmoid(h')) = sigma(dy'*W)
    cae.dh = zeros(size(cae.h));
	if opts.convapi
		cae.dh = vl_nnconv(cae.dy, cae.w, cae.w, 'pad', [0 1 0 1], 'stride', 1) ;
	else
		for pt = 1:para.bsze
			for oc = 1:cae.oc
				for ic = 1:cae.ic
					cae.dh(:,:,oc,pt) = cae.dh(:,:,oc,pt)+conv2(cae.dy(:,:,ic,pt),cae.w(:,:,ic,oc),'valid');
				end                   
			end        
		end
	end
    if cae.ps>=2        
        cae.dh = cae.dh.*cae.h_mask;
    end
    % dsigmoid(h')/dh'
    cae.dh = cae.dh.*(cae.h.*(1-cae.h)); 
    % dloss/db = sigma(dh')
    cae.db = reshape(sum(sum(cae.dh)),[size(cae.b) para.bsze]);
    % dloss/dw = x~*dh'+dy'~*h
    cae.dw = zeros([size(cae.w) para.bsze]);
    cae.dy_tilde = flip(flip(cae.dy,1),2);
    x_tilde = flip(flip(x,1),2);
    for pt = 1:para.bsze
        for oc = 1:cae.oc
            for ic = 1:cae.ic                
                % x~*dh+dy~*h, perfect                
                cae.dw(:,:,ic,oc,pt) = conv2(x_tilde(:,:,ic,pt),cae.dh(:,:,oc,pt),'valid')+conv2(cae.dy_tilde(:,:,ic,pt),cae.h_pool(:,:,oc,pt),'valid');
            end
        end        
    end    
    
    cae.dc = sum(cae.dc,3);
    cae.db = sum(cae.db,3);
    cae.dw = sum(cae.dw,5);    
end

function [cae] = cae_update(cae, opts)
    cae.b = cae.b - opts.alpha*cae.db;
    cae.c = cae.c - opts.alpha*cae.dc;
    cae.w = cae.w - opts.alpha*cae.dw;
    cae.w_tilde = flip(flip(cae.w,1),2);
end

function [numdw,numdb,numdc] = cae_check_grad(cae, x, para)
    epsilon = 1e-5;
    
    numdw = zeros(size(cae.dw));
    numdc = zeros(size(cae.dc));
    numdb = zeros(size(cae.db));
    
    % dc
    for ic = 1:cae.ic
        cae_h = cae;                    
        cae_h.c(ic) = cae_h.c(ic)+epsilon;
        cae_h = cae_ffbp(cae_h,x,para);

        cae_l = cae;
        cae_l.c(ic) = cae_l.c(ic)-epsilon;
        cae_l = cae_ffbp(cae_l,x,para);
        
        numdc(ic) = (cae_h.loss - cae_l.loss) / (2 * epsilon);
    end
    % db
    for oc = 1:cae.oc
        cae_h = cae;                    
        cae_h.b(oc) = cae_h.b(oc)+epsilon;
        cae_h = cae_ffbp(cae_h,x,para);

        cae_l = cae;
        cae_l.b(oc) = cae_l.b(oc)-epsilon;
        cae_l = cae_ffbp(cae_l,x,para);
        
        numdb(oc) = (cae_h.loss - cae_l.loss) / (2 * epsilon);
    end
    % dw
    for ic = 1:cae.ic
        for oc = 1:cae.oc
            for m = 1:cae.ks
                for n = 1:cae.ks
                    cae_h = cae;                            
                    cae_h.w(m,n,ic,oc) = cae_h.w(m,n,ic,oc)+epsilon;
                    cae_h.w_tilde(cae.ks+1-m,cae.ks+1-n,ic,oc) = cae_h.w_tilde(cae.ks+1-m,cae.ks+1-n,ic,oc)+epsilon;                                                     
                    cae_h = cae_ffbp(cae_h,x,para);
                    
                    cae_l = cae;
                    cae_l.w(m,n,ic,oc) = cae_l.w(m,n,ic,oc)-epsilon;
                    cae_l.w_tilde(cae.ks+1-m,cae.ks+1-n,ic,oc) = cae_l.w_tilde(cae.ks+1-m,cae.ks+1-n,ic,oc)-epsilon;    
                    cae_l = cae_ffbp(cae_l,x,para);
                    
                    numdw(m,n,ic,oc) = (cae_h.loss - cae_l.loss) / (2 * epsilon);
                end
            end           
        end
    end
end

function [x,para] = cae_check(cae, x, opts)

    para.m = size(x,1);
    para.pnum = size(x,4); % number of data points
    para.pgrds = (para.m-cae.ks+1)/cae.ps; % pool grids
    para.bsze = opts.batchsize; % batch size
    para.bnum = para.pnum/para.bsze; % number of batches
    
    if size(x,3)~=cae.ic
        error('number of input chanels doesn''t match');
    end
    
    if cae.ks>para.m
        error('too large kernel');
    end
    
    if floor(para.pgrds)~=para.pgrds
        error('sides of hidden representations should be divisible by pool size')
    end
    
    if floor(para.bnum)~=para.bnum
        error('number of data points should be divisible by batch size.');
    end
end
