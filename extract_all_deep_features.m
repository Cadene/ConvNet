clc;
clear all;
close all;

imgPath = '/usr/local/VOCdevkit/VOC2007/JPEGImages/';
listeImg = dir(imgPath);
listeImg = listeImg(3:size(listeImg));

% Charger le reseau VGG-M
run /usr/local/matconvnet-1.0-beta15/matlab/vl_setupnn
VggPath = '/usr/local/imagenet/';
net = load(strcat(VggPath,'imagenet-vgg-s.mat'));

% Matrice des features
nb_images = size(listeImg);
nb_images = nb_images(1);

tailles_feature = [4096 4096 4096 4096 1000 1000];
firstLayer = 15;
nbLayers = size(tailles_feature,2);

all_layers_features = {};

for h=1:nbLayers
    all_layers_features{h} = zeros(nb_images, tailles_feature(h));
end
tic
for i = 1:size(listeImg)      
    imgName = listeImg(i).name; 
	% Charger et préparer l'image
    im = imread(strcat(imgPath,imgName));
    im_ = single(im);
    im_ = imresize(im_, net.normalization.imageSize(1:2));
    im_ = im_ - net.normalization.averageImage;
    % output des couches:
    res = vl_simplenn(net, im_);
    for h=1:nbLayers
        all_layers_features{h}(i,:) = res(h+firstLayer).x;
    end
end
toc
for h=1:nbLayers
    filename = strcat('/Vrac/3152691/rdfia/features_cnns',int2str(firstLayer+h-1),'.mat');
    all_features = all_layers_features{h};
    save(filename, 'all_features');
end

% 
% % scores
% scores = squeeze(res(end).x);
% [bestScore, best] = max(scores);
% figure(1);
% clf;
% imagesc(im);
% title(sprintf('%s (%d), score %.3f', net.classes.description{best}, best, bestScore));
