function showcifar(info,varargin)
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
  title1 = {};t1 = {'���ۺ���'};
  title1 = horzcat(title1, strcat(t1, varargin)) ;
  title(title1{:}) ;
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
     title2 = {};t2 = {'������'};
  title2 = horzcat(title2, strcat(t2, varargin)) ;
  title(title2{:}) ;
end