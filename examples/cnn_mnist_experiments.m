%% Experiment with the cnn_mnist_fc_bnorm
ts=tic;
[net_mnist_bn, info_mnist_bn] = cnn_mnist(...
  'expDir', 'data/mnist-bnorm', 'useBnorm', true);
mnist_time_bn = toc(ts);
ts2=tic;
[net_mnist_fc, info_mnist_fc] = cnn_mnist(...
  'expDir', 'data/mnist-baseline', 'useBnorm', false);
mnist_time_fc = toc(ts2);

%%
figure(1) ; clf ;
subplot(1,2,1) ;
semilogy(info_mnist_fc.val.objective, '.-') ; hold on ;
semilogy(info_mnist_bn.val.objective, '.--') ;
xlabel('训练次数'); ylabel('代价函数') ;
grid on ;
h=legend('MNIST', 'MNIST_BN') ;
set(h,'color','none');
title('代价函数') ;
subplot(1,2,2) ;
plot(info_mnist_fc.val.error(1,:), '.-') ; hold on ;
plot(info_mnist_fc.val.error(2,:), '.--') ;
plot(info_mnist_bn.val.error(1,:), '.-') ;
plot(info_mnist_bn.val.error(2,:), '.--') ;
h=legend('MNIST-val','MNIST-val-3','MNIST_BN-val','MNIST_BN-val-3') ;
grid on ;
xlabel('训练次数'); ylabel('错误率') ;
set(h,'color','none') ;
title('错误率') ;
drawnow ;
save mnist_params_fixed.mat