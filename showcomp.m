function showcomp(info_cifar_fc,info_cifar_bn,lr1,lr2)
figure(1) ; clf ;
subplot(1,2,1) ;
semilogy(info_cifar_fc.train.objective, '.-') ; hold on ;
semilogy(info_cifar_bn.train.objective, '.--') ;
xlabel('训练次数'); ylabel('代价函数') ;
grid on ;
leg = {};
sig = {'CIFAR', 'CIFAR-BN'};
% sig = {'CIFAR-BN', 'CIFAR-BN'};
leg = horzcat(leg, strcat(sig{1}, lr1)) ;
leg = horzcat(leg, strcat(sig{2}, lr2)) ;
h=legend(leg{:}) ;
set(h,'color','none');
title('代价函数') ;
subplot(1,2,2) ;
plot(info_cifar_fc.val.error(1,:), '.-') ; hold on ;
plot(info_cifar_fc.val.error(2,:), '.--') ;
plot(info_cifar_bn.val.error(1,:), '.-') ;
plot(info_cifar_bn.val.error(2,:), '.--') ;

sig2={'CIFAR-val','CIFAR-val-3','CIFAR-BN-val','CIFAR-BN-val-3'};
% sig2={'CIFAR-BN-val','CIFAR-BN-val-3','CIFAR-BN-val','CIFAR-BN-val-3'};
leg2={};
leg2 = horzcat(leg2, strcat(sig2{1}, lr1)) ;
leg2 = horzcat(leg2, strcat(sig2{2}, lr1)) ;
leg2 = horzcat(leg2, strcat(sig2{3}, lr2)) ;
leg2 = horzcat(leg2, strcat(sig2{4}, lr2)) ;
h=legend(leg2{:}) ;
grid on ;
xlabel('训练次数'); ylabel('错误率') ;
set(h,'color','none') ;
title('错误率') ;
drawnow ;

end

