%% Script/instructions on how to submit plots/answers for question 2.
% Put your textual answers where specified in this script and run it before
% submitting.

% Loading the data: this loads X, Xnoisy, and Y.
load('../data/breast-cancer-data-fixed.mat');

%% 2.1
answers{1} = 'The noisy dataset has higher means and lower std deviations as compared to the regular dataset. This holds in case of all four values of N_folds. The test error is less than the knn error in case of both regular and noisy data.';

% Plotting with error bars: first, arrange your data in a matrix as
% follows:
%
%  nfold_errs(i,j) = nfold error with n=j of i'th repeat
%  
% Then we want to plot the mean with error bars of standard deviation as
% folows: y = mean(nfold_errs), e = std(nfold_errs), x = [2 4 8 16].
% 
% >> errorbar(x, y, e);
%
% Along with nfold_errs, also plot errorbar for test error. This will 
% serve as measure of performance for different nfold-crossvalidation.
%
% To add labels to the graph, use xlabel('X axis label') and ylabel
% commands. To add a title, using the title('My title') command.
% See the class Matlab tutorial wiki for more plotting help.
% 
% Once your plot is ready, save your plot to a jpg by selecting the figure
% window and running the command:
%
% >> print -djpg plot_2.1-noisy.jpg % (for noisy version of data)
% >> print -djpg plot_2.1.jpg  % (for regular version of data)
%
K = 1;
N_folds_vec = [2,4,8,16];
distFunc='l2';
knnerr=zeros(100,16);
knnerr_noisy=zeros(100,1);
for i=1:100
    r = randperm(size(X,1));
    testerror=0;
    testerror_noisy=0;
    train_indices = r(1:400);
    trainPoints=X(train_indices,:);
    trainPoints_noisy=X_noisy(train_indices,:);
    trainLabels=Y(train_indices);
    test_indices = r(401:683);
    testPoints = X(test_indices,:);
    testPoints_noisy=X_noisy(test_indices,:);
    testLabels = Y(test_indices);
    for k=1:size(N_folds_vec,2)
                N_folds=N_folds_vec(k);
                
                part=make_xval_partition(400,N_folds);
                knnerror=knn_xval_error(K,trainPoints,trainLabels,part,distFunc);
                knnerror_noisy=knn_xval_error(K,trainPoints_noisy,trainLabels,part,distFunc);
                knnerr(i,N_folds_vec(1,k))=knnerror;
                knnerr_noisy(i,N_folds_vec(1,k))=knnerror_noisy;
    end
    %Computing the test error
    estimatedTestLabels = knn_test(K,trainPoints,trainLabels,testPoints,distFunc);
    estimatedTestLabels_noisy = knn_test(K,trainPoints_noisy,trainLabels,testPoints_noisy,distFunc);    
    testerror=testerror+sum(testLabels.*estimatedTestLabels<0)/size(testLabels,1);
    testerror_noisy=testerror_noisy+sum(testLabels.*estimatedTestLabels_noisy<0)/size(testLabels,1);
end
    x= 1:16;
    %knn , regular
    y= mean(knnerr);
    e= std(knnerr);
    errorbar(x,y,e);
    hold on;
    x=2;
    %test , regular
    y= mean(testerror);
    e= std(testerror);
    errorbar(x,y,e);
    hold on;
    x=4;
    %test , regular
    y= mean(testerror);
    e= std(testerror);
    errorbar(x,y,e);
    hold on;
    x=8;
    %test , regular
    y= mean(testerror);
    e= std(testerror);
    errorbar(x,y,e);
    hold on;
    x=16;
    %test , regular
    y= mean(testerror);
    e= std(testerror);
    errorbar(x,y,e);
    hold on;
    

    xlabel('Number of folds')
    xlabel('Error')
    title('N-Fold error of regular dataset (2.1)')
    print -djpeg plot_2.1.jpg
    hold off;
   
    x=1:16;
    %knn, noisy
    y= mean(knnerr_noisy);
    e= std(knnerr_noisy);
    errorbar(x,y,e);
    hold on;
    
    %test , noisy
    x=2;
    y= mean(testerror_noisy);
    e= std(testerror_noisy);
    errorbar(x,y,e);

    xlabel('Number of folds')
    ylabel('Error')
    title('N-Fold error of noisy dataset (2.1)')
    print -djpeg plot_2.1-noisy.jpg
    hold off;
   

%% 2.2
answers{2} = 'k=2,sigma=4 are the values of k and sigma corresponding to which the test error is minimum in case of a regular dataset. k=2,sigma=6 are the values of k and sigma corresponding to which the test error is minimum in case of a noisy dataset.';

% Save your plots as follows:
%
%  noisy data, k-nn error vs. K --> plot_2.2-k-noisy.jpg
%  noisy data, kernreg error vs. sigma --> plot_2.2-sigma-noisy.jpg
%  regular data, k-nn error vs. K --> plot_2.2-k.jpg
%  regular data, kernreg error vs. sigma --> plot_2.2-sigma.jpg

% K = {1,3,5,...,15}
% sigma = {1,2,3,...,8}
% Run with 10 folds
N=10;
distFunc='l2';
knnerr=zeros(100,15);
knnerr_noisy=zeros(100,15);
knnerr_test=zeros(100,15);
knnerr_test_noisy=zeros(100,15);
kernerr=zeros(100,8);
kernerr_noisy=zeros(100,8);
kernerr_test=zeros(100,8);
kernerr_test_noisy=zeros(100,8);

for i = 1:100
    r = randperm(size(X,1));
    %knnerr_test=zeros(100,15);  
    %knnerr_test_noisy=zeros(100,15);
    %kernerr_test=zeros(100,8);
    %kernerr_test_noisy=zeros(100,8);
    train_indices = r(1:400);
    trainPoints=X(train_indices,:);
    trainPoints_noisy=X_noisy(train_indices,:);
    trainLabels=Y(train_indices);
    test_indices = r(401:683);
    testPoints = X(test_indices,:);
    testPoints_noisy=X_noisy(test_indices,:);
    testLabels = Y(test_indices);
    for K=1:15
               
                part=make_xval_partition(400,10);
                knnerror=knn_xval_error(K,trainPoints,trainLabels,part,distFunc);
                knnerror_noisy=knn_xval_error(K,trainPoints_noisy,trainLabels,part,distFunc);
                knnerr(i,K)=knnerror;
                knnerr_noisy(i,K)=knnerror_noisy;
                estimatedTestLabels = knn_test(K,trainPoints,trainLabels,testPoints,distFunc);
                estimatedTestLabels_noisy = knn_test(K,trainPoints_noisy,trainLabels,testPoints_noisy,distFunc);    
                knnerr_test(i,K)=knnerr_test(i,K)+sum(testLabels.*estimatedTestLabels<0)/size(testLabels,1);
                knnerr_test_noisy(i,K)=knnerr_test_noisy(i,K)+sum(testLabels.*estimatedTestLabels_noisy<0)/size(testLabels,1);
    end
    %Computing the test error
    
    
    for sigma=1:8
               
                part=make_xval_partition(400,10);
                kernregerror=kernreg_xval_error(sigma,trainPoints,trainLabels,part,distFunc);
                kernregerror_noisy=kernreg_xval_error(sigma,trainPoints_noisy,trainLabels,part,distFunc);
                kernerr(i,sigma)=kernregerror;
                kernerr_noisy(i,sigma)=kernregerror_noisy;
                estimatedTestLabels = kernreg_test(sigma,trainPoints,trainLabels,testPoints,distFunc);
                estimatedTestLabels_noisy = kernreg_test(sigma,trainPoints_noisy,trainLabels,testPoints_noisy,distFunc);    
                kernerr_test(i,sigma)=kernerr_test(i,sigma)+sum(testLabels.*estimatedTestLabels<0)/size(testLabels,1);
                kernerr_test_noisy(i,sigma)=kernerr_test_noisy(i,sigma)+sum(testLabels.*estimatedTestLabels_noisy<0)/size(testLabels,1);
    end
    %Computing the test error
    
end 

% Plots for k-nn, standard
x = 1:15;
e = std(knnerr);
y = mean(knnerr);
h1 = errorbar(x, y, e, 'b');

hold on;

x_test = 1:15;
e_test = std(knnerr_test);
y_test = mean(knnerr_test);
h2 = errorbar(x_test, y_test, e_test, 'r');

legend([h1 h2], 'Nfold error', 'Test error');

xlabel('K');
title('10-Fold error of Standard dataset using knn (2.2)');
print -djpeg plot_2.2-k.jpg

hold off;

% Plots for k-nn, noisy
x = 1:15;
e = std(knnerr_noisy);
y = mean(knnerr_noisy);
h1 = errorbar(x, y, e, 'b');

hold on;

x_test = 1:15;
e_test = std(knnerr_test_noisy);
y_test = mean(knnerr_test_noisy);
h2 = errorbar(x_test, y_test, e_test, 'r');

legend([h1 h2], 'Nfold error', 'Test error');

xlabel('K');
title('10-Fold error of Noisy dataset using k-nn (2.2)');
print -djpeg plot_2.2-k-noisy.jpg

hold off;

% Plots for kern, standard
x = 1:8;
e = std(kernerr);
y = mean(kernerr);
h1 = errorbar(x, y, e, 'b');

hold on;

x_test = 1:8;
e_test = std(kernerr_test);
y_test = mean(kernerr_test);
h2 = errorbar(x_test, y_test, e_test, 'r');

legend([h1 h2], 'Nfold error', 'Test error');

xlabel('Sigma');
title('10-Fold error of Standard dataset using kernreg (2.2)');
print -djpeg plot_2.2-sigma.jpg

hold off;

% Plots for kern, noisy
x = 1:8;
e = std(kernerr_noisy);
y = mean(kernerr_noisy);
h1 = errorbar(x, y, e, 'b');

hold on;

x_test = 1:8;
e_test = std(kernerr_test_noisy);
y_test = mean(kernerr_test_noisy);
h2 = errorbar(x_test, y_test, e_test, 'r');

legend([h1 h2], 'Nfold error', 'Test error');

xlabel('Sigma');
title('10-Fold error of Noisy dataset using kernreg (2.2)');
print -djpeg plot_2.2-sigma-noisy.jpg

hold off;

%% Finishing up - make sure to run this before you submit.
save('problem_2_answers.mat', 'answers');
