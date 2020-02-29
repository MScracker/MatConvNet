function [] = showres(res)
	for l = 1:numel(res)
		fprintf('Layers %d: %s\n',l,num2str(size(res(l).x)));
	end
end