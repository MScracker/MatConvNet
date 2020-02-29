function [] = shownet(net)
	for i = 1:numel(net.layers)
		fprintf('Layers{%d}:',i);
        net.layers{i}
	end
end