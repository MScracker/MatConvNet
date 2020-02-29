% ts1=tic;
% [net_cifar_fc, info_cifar_fc] = cnn_cifar(...
%   'expDir', 'new/cifar-fc','imdbPath','new/cifar-fc/cifar-nowhite.mat','isMean',false,'whitenData',false ,'useBnorm', false,...
%   'learningRate',[0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,5)]);
% cifar_time_fc = toc(ts1);
% 
% ts2=tic;
% [net_cifar_bn, info_cifar_bn] = cnn_cifar(...
%   'expDir', 'new/cifar-bn','imdbPath','new/cifar-bn/cifar-nowhite.mat','isMean',false,'whitenData',false ,'useBnorm', true,...
%   'learningRate',[0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,5)]);
% cifar_time_bn = toc(ts2);
% 
% ts3=tic;
% [net_cifar_bn2, info_cifar_bn2] = cnn_cifar(...
%   'expDir', 'new/cifar-bn2','imdbPath','new/cifar-bn2/cifar-nowhite.mat','isMean',false,'whitenData',false ,'useBnorm', true,...
%   'learningRate',[0.2*ones(1,15) 0.02*ones(1,10) 0.002*ones(1,5)]);
% cifar_time_bn2 = toc(ts3);
% 
% tc=tic;
% [net_mnist_fc, info_mnist_fc] = cnn_mnist(...
%   'expDir', 'new/mnist-fc', 'imdbPath','new/mnist-fc/mnist-nomean.mat','isMean',false,'sigmoid',true,'useBnorm', false,'learningRate',0.001);
% mnist_time_fc = toc(tc);

% tc2=tic;
% [net_mnist_bn, info_mnist_bn] = cnn_mnist(...
%   'expDir', 'new/mnist-bn','imdbPath','new/mnist-bn/mnist-nomean.mat','isMean',false,'sigmoid',true, 'useBnorm', true,'learningRate',0.001);
% mnist_time_bn = toc(tc2);

tc3=tic;
[net_mnist_bn2, info_mnist_bn2] = cnn_mnist(...
  'expDir', 'new/mnist-bn2','imdbPath','new/mnist-bn2/mnist-nomean.mat','isMean',false, 'sigmoid',true,'useBnorm', true,'learningRate',0.005);
mnist_time_bn2 = toc(tc3);

tc4=tic;
[net_mnist_bn4, info_mnist_bn4] = cnn_mnist(...
  'expDir', 'new/mnist-bn4','imdbPath','new/mnist-bn4/mnist-nomean.mat','isMean',false, 'useBnorm', true,'learningRate',0.001);
mnist_time_bn4 = toc(tc4);


ts_fc=tic;
[net_cifar_fc2, info_cifar_fc2] = cnn_cifar(...
  'expDir', 'new/cifar-fc2','imdbPath','new/cifar-fc2/cifar-nowhite.mat','isMean',true,'whitenData',false ,'useBnorm', false,...
  'learningRate',[0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,5)]);
cifar_time_fc2 = toc(ts_fc);

ts4=tic;
[net_cifar_bn4, info_cifar_bn4] = cnn_cifar(...
  'expDir', 'new/cifar-bn4','imdbPath','new/cifar-bn4/cifar-nowhite.mat','isMean',true,'sigmoid',true,'whitenData',false ,'useBnorm', true,...
  'learningRate',[0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,5)]);
cifar_time_bn4 = toc(ts2);

save ('new/newexp.mat'); 