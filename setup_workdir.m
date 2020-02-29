function setup_workdir(varargin)
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'cifar'));
addpath(genpath(fullfile(root, 'cifar')));
addpath(fullfile(root, 'data'));
addpath(genpath(fullfile(root, 'stl10')));
end