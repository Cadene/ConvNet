clc;
clear all;
close all;

imgPath = '/usr/local/VOCdevkit/VOC2007/JPEGImages/'
listeImg = dir(imgPath);
listeImg = listeImg(3:size(listeImg));

% Charger le reseau VGG-M
run /usr/local/matconvnet-1.0-beta15/matlab/vl_setupnn
VggPath = '/usr/local/imagenet/';
net = load(strcat(VggPath,'imagenet-vgg-m.mat'));

% Matrice des features
nb_images = size(listeImg);
nb_images = nb_images(1);
taille_feature = 4096;
all_features = zeros(nb_images, taille_feature);

for i = 1:size(listeImg)      
    imgName = listeImg(i).name;
	% Charger et préparer l'image
    im = imread(strcat(imgPath,imgName));
    im_ = single(im);
    im_ = imresize(im_, net.normalization.imageSize(1:2));
    im_ = im_ - net.normalization.averageImage;
    % output des couches:
    res = vl_simplenn(net, im_);
    all_features(i,:) = res(20).x;
end

save features_cnnm20.mat all_features

% 
% % scores
% scores = squeeze(res(end).x);
% [bestScore, best] = max(scores);
% figure(1);
% clf;
% imagesc(im);
% title(sprintf('%s (%d), score %.3f', net.classes.description{best}, best, bestScore));
