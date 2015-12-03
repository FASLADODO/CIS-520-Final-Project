%% Script/instructions on how to submit plots/answers for question 3.
% Put your textual answers where specified in this script and run it before
% submitting.

% Loading the data
 data = load('../data/mnist_all.mat');

% Running a training set for binary decision tree classifier
[Xtrain Ytrain] = get_digit_dataset(data, {'7','9'}, 'train');

%% Train a depth 4 binary decision tree
dectree = dt_train(Xtrain, Ytrain, 4);

%%
[Xtest Ytest] = get_digit_dataset(data, {'7','9'}, 'test');
Yhat = zeros(size(Ytest));

for i = 1:size(Xtest,1)
    Yhat(i) = dt_value(dectree, Xtest(i,:)) >= 0.5;
end
mean(Yhat ~= Ytest)

%% 3.1
answers{1} = 'Overfitting is observed at x=6 as the testing error becomes more than the training error.';

% Saving your plot: once you have succesfully plotted your data; e.g.,
% something like:
% >> plot(depth, [train_err test_err]);
% Remember: You can save your figure to a .jpg file as follows:
% >> print -djpg plot_3.1.jpg
dep=6;

train_err = zeros(1,dep);
test_err = zeros(1,dep);
depth = 1:dep;

for d = depth
    [Xtrain Ytrain] = get_digit_dataset(data, {'1','3','7'}, 'train');
    dectree = dt_train_multi(Xtrain, Ytrain, d);
    [Xtest Ytest] = get_digit_dataset(data, {'1','3','7'}, 'test');
    Yhat_train = zeros(size(Ytrain));
    Yhat_test = zeros(size(Ytest));
    
    % Get training and test errors
    for i = 1:size(Xtrain,1)
        value = dt_value(dectree, Xtrain(i,:));
        [label, ind] = max(value);
        Yhat_train(i) = ind;
    end
    train_err(d) = mean(Yhat_train ~= Ytrain);
    
    for i = 1:size(Xtest,1)
        value = dt_value(dectree, Xtest(i,:));
        [label, ind] = max(value);
        Yhat_test(i) = ind;
    end
    test_err(d) = mean(Yhat_test ~= Ytest);
end

plot(depth, train_err,'.b')
hold on;
plot(depth, test_err, '*r')
xlabel('Depth of the tree')
legend([plot(depth, train_err,'.b') plot(depth, test_err, '*r')], 'Training error', 'Test error');
title('Plot of errors vs. depth of the tree')
print -djpeg plot_3.1.jpg
hold off;

%% 3.2
answers{2} = 'As the confusion matrix clearly shows the digit 3 is commonly confused for a 5.';

% Saving your plot: once you've computed M, plot M with the plotnumeric.m
% command we've provided. e.g:
% >> plotnumeric(M);
%
% Save your file to plot_3.2.jpg
%
% ***** ALSO *******
% Save your confusion matrix M to a .txt file as follows:
% >> save -ascii confusion.txt M

% Obtaining training and testing data
[Xtrain, Ytrain] = get_digit_dataset(data, {'0','1','2','3','4','5','6','7','8','9'}, 'train');
dectree = dt_train_multi(Xtrain, Ytrain, 6);
[Xtest, Ytest] = get_digit_dataset(data, {'0','1','2','3','4','5','6','7','8','9'}, 'test');
Ytest = Ytest - 1; %subtracting 1 to get 0:9 instead of 1:10
Yhat = zeros(size(Ytest));
M=zeros(10);

for i = 1:size(Xtest,1)
    val = dt_value(dectree, Xtest(i,:));
    [label, ind] = max(val);
    Yhat(i) = ind-1; %subtracting 1 to get 0:9 instead of 1:10
end


for i=1:size(Yhat,1)
    m=Ytest(i)+1; %Since Ytest belongs to 0:9 but indices in MATLAB start from 1
    n=Yhat(i)+1;
    M(m,n)=(M(m,n)+1);
end        
M=M./numel(Ytest);

plotnumeric(M);
xlabel('Estimated digit + 1');
ylabel('Original digit + 1');
title('Confusion matrix ');
save -ascii confusion.txt M
print -djpeg plot_3.2.jpg



%% 3.3
answers{3} = 'Considering the misclassification of 3 as 8 . As the pixel representation indicates, the two digits look similar';

% E.g., if Xtest(i,:) is an example your method fails on, call:
% >> plot_dt_digit(tree, Xtest(i,:));
%
% Save your file to plot_3.3.jpg

sub=0;
for i = 1:size(Yhat,1)
    m = Ytest(i)+1;
    n = Yhat(i)+1;
    if m == 4 & n == 9 
        if sub==0
           sub = i; 
        end
    end
end
H=plot_dt_digit(dectree, Xtest(sub,:));
print -djpeg plot_3.3.jpg

%% Finishing up - make sure to run this before you submit.
save('problem_3_answers.mat', 'answers');