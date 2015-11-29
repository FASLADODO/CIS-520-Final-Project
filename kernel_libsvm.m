x1=new_features;
x2=train_new_image_features;

x1_test=test_new_features;
x2_test=test_new_image_features;

X=[x1,x2];
X_test=[x1_test,x2_test];
    
kIntersect = @(x1,x2) kernel_intersection(x1, x2);

K = kIntersect(X, X);
Ktest = kIntersect(X, X_test);
Ytest = zeros(size(X_test,1),1);
        
%Use built-in libsvm cross validation to choose the C regularization
%parameter.
crange = 10.^[-10:2:3];
for i = 1:numel(crange)
    acc(i) = svmtrain(genders, [(1:size(K,1))' K], sprintf('-t 4 -v 10 -c %g', crange(i)));
end
[~, bestc] = max(acc);
fprintf('Cross-val chose best C = %g\n', crange(bestc));

%Train and evaluate SVM classifier using libsvm
model = svmtrain(genders, [(1:size(K,1))' K], sprintf('-t 4 -c %g', crange(bestc)));
[yhat acc vals] = svmpredict(Ytest, [(1:size(Ktest,1))' Ktest], model);
