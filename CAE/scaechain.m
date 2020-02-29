function [ scae ] = scaechain( scae, x, res,varargin)
%CAECHAIN Summary of this function goes here
%   Detailed explanation goes here
    opts.backprop = true;
    opts.cudnn = true ;
    opts = vl_argparse(opts,varargin);
    if opts.cudnn
        cudnn = {'CuDNN'} ;
    else
        cudnn = {'NoCuDNN'} ;   
    end
    n = numel(scae.layer);
  
    if nargin <= 2 || isempty(res)
        res = struct(...
        'x', cell(1,n+1), ...
        'dzdx', cell(1,n+1), ...
        'dzdw', cell(1,n+1), ...
        'aux', cell(1,n+1), ...
        'time', num2cell(zeros(1,n+1)), ...
        'backwardTime', num2cell(zeros(1,n+1))) ;
    end
    res(1).x = x ;
    
    for i=1:n
        node = scae.layer{i};
        res(i).time = tic ;
        switch node.type
            case 'cae_up'
                %res.x here impiles h;
                res(i+1).x = vl_nnconv(res(i).x, node.w, node.b, ...
                               'pad', node.pad, 'stride', node.stride, ...
                               cudnn{:}) ;
            case 'cae_pool'
                %res.x here implies h_pool; res.aux here implies h_mask;
                [res(i+1).x , res(i+1).aux]= cae_pool(res(i).x, node.pool);                       
            case 'cae_down'
                res(i+1).x = cae_down(res(i).x, node.w, node.b); 
            case 'loss'
                res(i+1).x = cae_loss();
        end
        res(i).time = toc(res(i).time) ;
    end
    
    if opts.backprop
   
        for i=n:-1:1
             node = scae.layer{i};
             res(i).backwardTime = tic ;
             switch node.type
                 case 'bp_down'
                     
                     
             end    
             res(i).backwardTime = toc(res(i).backwardTime) ;
        end
    end
end
function [h_pool, h_mask] = cae_pool(x, pool)
    % ps: pool size
    if pool>=2
        h_pool = zeros(size(x));
        h_mask = zeros(size(x));
        x_height = size(x,1);
        y_height = size(x,2);
        for i = 1:x_height/pool
            for j = 1:y_height/pool
                grid = x((i-1)*pool+1:i*pool,(j-1)*pool+1:j*pool,:,:);           
                mx = repmat(max(max(grid)),pool,pool);
                mask = (grid==mx);
                sparse_grid = zeros(size(grid));
                sparse_grid(mask) = grid(mask);
                h_pool((i-1)*pool+1:i*pool,(j-1)*pool+1:j*pool,:,:) = sparse_grid;
                h_mask((i-1)*pool+1:i*pool,(j-1)*pool+1:j*pool,:,:) = mask;
            end
        end        
    end
end

function [out] = cae_down(x, w_tilde, b)
    % ks: kernel size, oc: output channels
    out = zeros(size(x));
    for pt = 1:size(x,4)
        for ic = 1:size(x,3)
            for oc = 1:size(w_tilde,4)
                out(:,:,ic,pt) = out(:,:,ic,pt) + conv2(x(:,:,oc,pt),w_tilde(:,:,ic,oc),'full');
            end
            out(:,:,ic,pt) = sigm(out(:,:,ic,pt)+b(ic));
        end        
    end
end

function [ loss ] = cae_loss(out, data)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    err = (out - data);
    loss = 1/2 * sum(err(:) .^2 )/size(data,4);
end


function [  ] = bp_down(out, data)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    cae.dy = cae.err.*(cae.o.*(1-cae.o))/para.bsze;
    cae.dc = reshape(sum(sum(cae.dy)),[size(cae.c) para.bsze]);
    
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
