function [error] = kernreg_xval_error(sigma, X, Y, parts, distFunc)
% KERNREG_XVAL_ERROR - Kernel regression cross-validation error.
%
% Usage:
%
%   ERROR = kernreg_xval_error(SIGMA, X, Y, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the kernel regression
% algorithm on the given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used (see KERNREG_TEST).
%
% Note that N = max(PART).
% SEE ALSO
%   MAKE_XVAL_PARTITION, KERNREG_TEST
    %Perform 100 iterations
    %define trainpoints and testpoints(400 in training, rest in testing)
    %invoke kernreg_test
    %calculate error between test labels obtained using kernreg_test and actual values
N_fold = max(parts);
error = 0;
for i=1:N_fold
    testPointIndex=zeros(size(X,1),1);
    %Creating an array of indices corresponding to which the part number is i 
    for j=1:size(X,1)
        if(parts(j)==i)
            testPointIndex(j,1)=1;
        end
    end
    testPointX = X( find(testPointIndex==1),:);
    testPointY = Y(find(testPointIndex==1))
    trainPointX = X(find(testPointIndex~=1),:);
    trainPointY = Y(find(testPointIndex~=1));
    estimatedTestPointY=kernreg_test(sigma, trainPointX, trainPointY, testPointX, distFunc);
    error=error+mean(testPointY.*estimatedTestPointY<0);

end
error=(error/N_fold);