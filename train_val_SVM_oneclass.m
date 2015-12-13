function [ best_c ] = train_val_SVM_oneclass( all_features, class )

labelPath = '/usr/local/VOCdevkit/VOC2007/ImageSets/Main/';

% Train ------------------------------------------------------

% Chargement des fichiers de train
[train_ids, train_labels] = textread(strcat(labelPath,class,'_train.txt'), '%d %d');

% Retirer les labels 0
labelsZero = (train_labels ~= 0);
train_ids = train_ids(labelsZero);
train_labels = train_labels(labelsZero);
train_features = sparse(all_features(train_ids,:));

% Entrainement du modèle
list_c = [10 1 .1 .01 .001 .0001 .00001];

[test_ids, test_labels] = textread(strcat(labelPath,class,'_val.txt'), '%d %d');
labelsZero = (test_labels ~= 0);
test_ids = test_ids(labelsZero);
test_labels = test_labels(labelsZero);
test_features = sparse(all_features(test_ids,:));

list_ap = zeros();
list_accuracy = zeros();

fileID = fopen('cval.csv', 'a');

s = size(list_c);
nb_c = s(2);

for i=1:nb_c
    c = list_c(i);
    model = train(train_labels, train_features, sprintf(' -c %f', c));
    [predicted_labels, accuracy, scores] = predict(test_labels, test_features, model);
    list_ap(i) = compute_class_AP(test_labels, scores);
    list_accuracy(i) = accuracy(1);
    fprintf(fileID, '%s %f %f %f\n', class, c, list_ap(i), list_accuracy(i));
end

[m, argm] = max(list_ap);
best_c = list_c(argm);

fclose(fileID);

