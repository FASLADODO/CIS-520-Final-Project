function [error] = knn_xval_error(K, X, Y, part, distFunc)
% KNN_XVAL_ERROR - KNN cross-validation error.
%
% Usage:
%
%   ERROR = knn_xval_error(K, X, Y, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the K-NN algorithm on the 
% given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used (see KNN_TEST).
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, KNN_TEST

N_fold = max(part);
error = 0;
for i=1:N_fold
    testPointIndex=zeros(size(X,1),1);
    %Creating an array of indices corresponding to which the part number is i 
    for j=1:size(X,1)
        if(part(j)==i)
            testPointIndex(j,1)=1;
        end
    end
    testPointX = X( find(testPointIndex==1),:);
    testPointY = Y(find(testPointIndex==1));
    trainPointX = X(find(testPointIndex~=1),:);
    trainPointY = Y(find(testPointIndex~=1));
    estimatedTestPointY=knn_test(K, trainPointX, trainPointY, testPointX, distFunc);
    error=error+mean(testPointY.*estimatedTestPointY<0);
end
error=(error/N_fold);