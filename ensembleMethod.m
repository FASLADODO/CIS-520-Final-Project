% sumTrain = sum(words_train,2);
% sumTest = sum(words_test,2);
% genders=genders_train;
% 
% avgTrain = words_train./repmat(sumTrain,[1,5000]);
% avgTest = words_test./repmat(sumTest,[1,5000]);
% 
% [~,score,~,~,explainedVar] = pca([avgTrain;avgTest]);
% cumulativeVar=cumsum(explainedVar);
% pcaComp = min(find(cumulativeVar>=97));

% Mdl_withoutPCAada = fitensemble(avgTrain, genders, 'AdaBoostM1',1000,'Discriminant', 'kFold',10, 'LearnRate', 0.1);
% loss_withoutPCAada = kfoldLoss(Mdl_withoutPCAada);

% Mdl_withoutPCAlogit = fitensemble(avgTrain, genders, 'LogitBoost',1000,'Discriminant', 'kFold',10, 'LearnRate', 0.1);
% loss_withoutPCAlogit = kfoldLoss(Mdl_withoutPCAlogit);

Mdl_withoutPCAnAvgAda = fitensemble(words_train, genders, 'AdaBoostM1',1000,'Discriminant', 'kFold',10, 'LearnRate', 0.1);
loss_withoutPCAnAvgAda = kfoldLoss(Mdl_withoutPCAnAvgAda);

% Mdl_withoutPCAnAvgLogit = fitensemble(words_train, genders, 'LogitBoost',1000,'Discriminant', 'kFold',10, 'LearnRate', 0.1);
% loss_withoutPCAnAvgLogit = kfoldLoss(Mdl_withoutPCAnAvgLogit);

pcaTrain = score(1:size(words_train,1),1:pcaComp);
pcaTest = score(size(words_train,1)+1:end,1:pcaComp);

Mdl_ada = fitensemble(pcaTrain, genders, 'AdaBoostM1',1000,'Discriminant', 'kFold',10, 'LearnRate', 0.1);
loss_ada = kfoldLoss(Mdl_ada);

Mdl_logit = fitensemble(pcaTrain, genders, 'LogitBoost',1000,'Discriminant', 'kFold',10, 'LearnRate', 0.1);
loss_logit = kfoldLoss(Mdl_logit);

Mdl_tree = fitctree(pcaTrain, genders, 'kFold', 10, 'LearnRate', 0.1);
loss_tree = kfoldLoss(Mdl_tree);

