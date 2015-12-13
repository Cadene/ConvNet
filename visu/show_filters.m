clc;
clear all;
close all;

% setup MtConvNet in MATLAB
run /usr/local/matconvnet-1.0-beta15/matlab/vl_setupnn

% Charger le reseau VGG-M
path = '/usr/local/imagenet/';
net = load(strcat(path,'imagenet-vgg-m.mat'));

filters = net.layers{1}.weights{1};
n_filters = size(filters,4);
size_filters = size(filters,1);

all_filters = zeros(8 * (size_filters+1),12 * (size_filters+1), 3);

i = 1;
j = 1;
for k=1:n_filters
    img = filters(:,:,:,k);
    img = (img / max(max(max(abs(img)))) + 1) / 2;
    
    all_filters(i:i+size_filters-1,j:j+size_filters-1,:) = img;
    
    j = j + size_filters + 2;
    if (j > 12*(size_filters+2))
        j = 1;
        i = i + size_filters + 2;
    end
end
imwrite(imresize(all_filters, 5), 'all_filter1.png');

% Charger et préparer l'image
%im = imread('peppers.png');
%im = imread('pears.png');
%im = imread('onion.png');
%im_ = single(im);
%im_ = imresize(im_, net.normalization.imageSize(1:2));
%im_ = im_ - net.normalization.averageImage;

% output des couches:
%res = vl_simplenn(net, im_);

% scores
%scores = squeeze(res(end).x);
%[bestScore, best] = max(scores);
%figure(1);
%clf;
%imagesc(im);
%title(sprintf('%s (%d), score %.3f', net.classes.description{best}, best, bestScore));

