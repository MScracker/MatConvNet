function [ ] = showim(res,layerth,imNum)
%SHOWLAYE Summary of this function goes here
%   Detailed explanation goes here
if nargin <= 2
    n = numel(res);
    if layerth > n
        error('exceeds the maximum layers.');
    else
        for i = 1: size(res(layerth).x,4)
            figure;colormap gray;
            vl_imarraysc(res(layerth).x(:,:,:,i),'spacing',1);
        end
    end
else
    
    n = numel(res);
    if layerth > n
        error('exceeds the maximum layers.');
    else
        figure;colormap gray;
        vl_imarraysc(res(layerth).x(:,:,:,imNum),'spacing',1);
    end
end
end

