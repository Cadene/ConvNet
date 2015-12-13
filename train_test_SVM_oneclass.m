function [ accuracy, ap ] = train_test_SVM_oneclass( all_features, class, c )

labelPath = '/usr/local/VOCdevkit/VOC2007/ImageSets/Main/';

% Train ------------------------------------------------------

% Chargement des fichiers de train
[train_ids, train_labels] = textread(strcat(labelPath,class,'_trainval.txt'), '%d %d');
%[train_ids, train_labels] = textread(strcat(labelPath,class,'_trainval.txt'));

% Retirer les labels 0
labelsZero = (train_labels ~= 0);
train_ids = train_ids(labelsZero);
train_labels = train_labels(labelsZero);
train_features = sparse(all_features(train_ids,:));

% Entrainement du modèle
model = train(train_labels, train_features, sprintf(' -c %f', c));
%model = train(train_labels, train_features);

% Test -------------------------------------------------------
% Chargement du fichier de test
[test_ids, test_labels] = textread(strcat(labelPath,class,'_test.txt'), '%d %d');
%[test_ids, test_labels] = textread(strcat(labelPath,class,'_test.txt'));

% Retirer les labels 0
labelsZero = (test_labels ~= 0);
test_ids = test_ids(labelsZero);
test_labels = test_labels(labelsZero);
test_features = sparse(all_features(test_ids,:));

% Evaluation
[predicted_labels, accuracy, scores] = predict(test_labels, test_features, model);
ap = compute_class_AP(test_labels, scores);
end
