function [ ss ] = testVar( a, x ,varargin)
%TESTVAR Summary of this function goes here
%   Detailed explanation goes here
   opt.a = true;
   opt = vl_argparse(opt,varargin);
    ss = a;
  if opt.bp
    ss = a+x;
  end

end

