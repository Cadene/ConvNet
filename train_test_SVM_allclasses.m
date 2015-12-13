clc;
clear all;
close all;

% SVM
%addpath /usr/local/liblinear-2.1/matlab/

% MAP
%addpath /usr/local/Code_TP/

c = 1;

featuresFile = 'features_cnnm19.mat';
load(featuresFile); % Récupère all_features

classes = {'aeroplane' 'bicycle' 'bird' 'boat' 'bottle' 'bus' 'car' 'cat' 'chair' 'cow' 'diningtable' 'dog' 'horse' 'motorbike' 'person' 'pottedplant' 'sheep' 'sofa' 'train' 'tvmonitor'};
labelPath = '/usr/local/VOCdevkit/VOC2007/ImageSets/Main/';

list_ap = zeros();
list_accuracy = zeros();

s = size(classes);
nbClass = s(2);

for i=1:nbClass
    cl = classes{i}
    [ accuracy, ap ] = train_test_SVM_oneclass(all_features, cl, c);
    list_accuracy(i) = accuracy(1);
    list_ap(i) = ap;
end

meanAccuracy = mean(list_accuracy);
list_ap
meanAP = mean(list_ap)