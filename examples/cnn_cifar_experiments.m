%% Experiment with the cnn_cifar_fc_bnorm
ts=tic;
[net_cifar_bn, info_cifar_bn] = cnn_cifar(...
  'expDir', 'data/cifar-bnorm', 'useBnorm', true);
time_bn = toc(ts);
ts2=tic;
[net_cifar_fc, info_cifar_fc] = cnn_cifar(...
  'expDir', 'data/cifar-baseline', 'useBnorm', false);
time_fc = toc(ts2);


%%
figure(1) ; clf ;
subplot(1,2,1) ;
semilogy(info_cifar_fc.val.objective, '.-') ; hold on ;
semilogy(info_cifar_bn.val.objective, '.--') ;
xlabel('训练次数'); ylabel('代价函数') ;
grid on ;
h=legend('CIFAR', 'CIFAR_BN') ;
set(h,'color','none');
title('代价函数') ;
subplot(1,2,2) ;
plot(info_cifar_fc.val.error(1,:), '.-') ; hold on ;
plot(info_cifar_fc.val.error(2,:), '.--') ;
plot(info_cifar_bn.val.error(1,:), '.-') ;
plot(info_cifar_bn.val.error(2,:), '.--') ;
h=legend('CIFAR-val','CIFAR-val-3','CIFAR_BN-val','CIFAR_BN-val-3') ;
grid on ;
xlabel('训练次数'); ylabel('错误率') ;
set(h,'color','none') ;
title('错误率') ;
drawnow ;

save cifar_params_fixed.mat