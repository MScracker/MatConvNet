function showcifar(info)
epoch =30;
errorLabels = {'top1e', 'top3e'} ;
figure
  subplot(1,2,1) ;
semilogy(1:epoch, info.train.objective, '.-', 'linewidth', 2) ;
    hold on ;
  semilogy(1:epoch, info.val.objective, '.--') ;
  xlabel('ѵ������') ; ylabel('���ۺ���') ;
  grid on ;
  h=legend('train','val') ;
  set(h,'color','none');
  title('���ۺ���') ;
  subplot(1,2,2) ; 
  leg = {};
  plot(1:epoch, info.train.error', '.-', 'linewidth', 2) ;
      hold on ;
      leg = horzcat(leg, strcat('train-', errorLabels)) ;
    plot(1:epoch, info.val.error', '.--') ;
    leg = horzcat(leg, strcat('val-', errorLabels)) ;
    set(legend(leg{:}),'color','none') ;
    grid on ;
    xlabel('ѵ������') ; ylabel('������') ;
    title('������');
end