n = size(data,1);
ns = floor(n/10);
for fold=1:10,
    if fold==1,
        testindices= ((fold-1)*ns+1):fold*ns;
        trainindices = fold*ns+1:n;
    else
        if fold==10,
            testindices= ((fold-1)*ns+1):n;
            trainindices = 1:(fold-1)*ns;
        else
            testindices= ((fold-1)*ns+1):fold*ns;
            trainindices = [1:(fold-1)*ns,fold*ns+1:n];
         end
    end
    % use testindices only for testing and train indices only for testing
    trainLabel = label(trainindices);
    trainData = data(trainindices,:);
    testLabel = label(testindices);
    testData = data(testindices,:)
    %# train one-against-all models
    model = cell(numLabels,1);
    for k=1:numLabels
        model{k} = svmtrain(double(trainLabel==k), trainData, '-c 1 -g 0.2 -b 1');
    end

    %# get probability estimates of test instances using each model
    prob = zeros(size(testData,1),numLabels);
    for k=1:numLabels
        [~,~,p] = svmpredict(double(testLabel==k), testData, model{k}, '-b 1');
        prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
    end

    %# predict the class with the highest probability
    [~,pred] = max(prob,[],2);
    acc = sum(pred == testLabel) ./ numel(testLabel);    %# accuracy
    C = confusionmat(testLabel, pred);                  %# confusion matrix
end