function DisplayFilter(filter, pad)
%% !Only Show the 1st-3rd channel of filter
 
%%
if nargin == 1
    % Between images padding
    pad = 1;
end
 
[M, N, P, Q] = size(filter);
width = ceil(sqrt(Q));
displayArray = zeros(pad + width * (M+pad), pad + width* (N+pad), 3);
 
 
count = 1;
for i = 1 : width
    for j =1 : width
        if count > Q
            break;
        end
                 
        f_ = filter(:, :, 1:3, count);
        absMinVal = abs(min(f_(:)));
        f_ = f_ + absMinVal;
        absMaxVal = max(abs(f_(:)));
        f_ = f_ / absMaxVal;
         
        displayArray(pad + (i - 1) * (M + pad) + (1 : M),...
                    pad + (j - 1) * (N + pad) + (1: N), :) ...
                    = f_;
        count = count + 1;
    end
end
 
imagesc(displayArray, [-1, 1]);
 
axis image off
drawnow;
end

