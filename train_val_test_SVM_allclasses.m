clc;
clear all;
close all;

% SVM
addpath /usr/local/liblinear-2.1/matlab/

% MAP
addpath /usr/local/Code_TP/

featuresFile = '/Vrac/3152691/rdfia/features_cnnm21.mat';
load(featuresFile); % Récupère all_features

classes = {'aeroplane' 'bicycle' 'bird' 'boat' 'bottle' 'bus' 'car' 'cat' 'chair' 'cow' 'diningtable' 'dog' 'horse' 'motorbike' 'person' 'pottedplant' 'sheep' 'sofa' 'train' 'tvmonitor'};
labelPath = '/usr/local/VOCdevkit/VOC2007/ImageSets/Main/';

list_ap = zeros();
list_accuracy = zeros();

s = size(classes);
nbClass = s(2);

tic

for i=1:nbClass
    cl = classes{i};
    c = train_val_SVM_oneclass(all_features, cl);
    [ accuracy, ap ] = train_test_SVM_oneclass(all_features, cl, c);
    list_accuracy(i) = accuracy(1);
    list_ap(i) = ap; 
end

toc

%meanAccuracy = mean(list_accuracy);
%fileID = fopen('AP_cnns.csv', 'a');
%fprintf(fileID, '%d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n', 20, list_ap);
%fclose(fileID);

meanAP = mean(list_ap)