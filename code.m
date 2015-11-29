%Dividing training data into six partitions
part = make_xval_partition(size(words_train,1), 6);
n=max(part);

for total=1:n
    a=1;
    b=1;

    for g=1:4998
        if part(g)==total
            words_test_final(a,:)=words_train(g,:);
            features_test_final(a,:)=features_train(g,:);
            ytest_final(a)=genders_train(g);
            a=a+1;
        else
            words_train_final(a,:)=words_train(g,:);
            features_train_final(a,:)=features_train(g,:);
            ytrain_final(a)=genders_train(g);
            b=b+1;
        end
    end
    svmstructwords = svmtrain(words_train_final, ytrain_final, 'Kernel_Function', 'polynomial', 'polyorder',2,'BoxConstraint', 0.2);
    testLabels_train_words = svmclassify(svmstructwords, words_test_final);

    svmstructimagefeatures = svmtrain(features_train_final, ytrain_final, 'Kernel_Function', 'polynomial', 'polyorder',2,'BoxConstraint', 0.2);
    testLabels_train_features = svmclassify(svmstructimagefeatures, features_test_final);

    combo = [testLabels_train_words, testLabels_train_features];
    ctree = fitctree(combo, ytest_final);
    combotest=[testwords,testimagefeatures];
end
    testLabels = predict(ctree, combotest);


