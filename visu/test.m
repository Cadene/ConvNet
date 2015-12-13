clc;
clear all;
close all;

% setup MtConvNet in MATLAB
run /usr/local/matconvnet-1.0-beta15/matlab/vl_setupnn

% Charger le reseau VGG-M
path = '/usr/local/imagenet/';
net = load(strcat(path,'imagenet-vgg-m.mat'));

% Charger et préparer l'image
%im = imread('peppers.png');
%im = imread('pears.png');
im = imread('onion.png');
im_ = single(im);
im_ = imresize(im_, net.normalization.imageSize(1:2));
im_ = im_ - net.normalization.averageImage;

% output des couches:
res = vl_simplenn(net, im_);

% scores
scores = squeeze(res(end).x);
[bestScore, best] = max(scores);
figure(1);
clf;
imagesc(im);
title(sprintf('%s (%d), score %.3f', net.classes.description{best}, best, bestScore));