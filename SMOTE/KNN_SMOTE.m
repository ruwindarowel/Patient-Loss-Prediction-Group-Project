%1st Indexed data is from the dataset that was not upsampled or downsampled
%2nd indexed data is from the datset that was downsampled
%Changing the age variable to categorical
%Now we can start fitting the KNN model
P=X_smote(:,2:25)';
T=y_smote';

rng(1234);
[trainV1,valV1,testV1]=dividevec(P,T,0.2,0.2);

trainV.P=trainV1.P';
valV.P=valV1.P';
testV.P=testV1.P';

trainV.T=trainV1.T';
valV.T=valV1.T';
testV.T=testV1.T';

%fitting KNN with defualt parameters
k1=fitcknn([trainV.P; valV.P],[trainV.T; valV.T],'CategoricalPredictors','all','Distance','hamming');

%calculate prediction accuracy of initial tree
resub_k1=resubLoss(k1);

Y_k1= predict(k1,testV.P);

errR_k1= sum(testV.T~= Y_k1)/length(testV.T)

%optimizing KNN to find opitimal k, using cross validation.
rng(1234);
errR = [];
for i = 1:25
    k11=fitcknn([trainV.P; valV.P],[trainV.T; valV.T],'CategoricalPredictors','all','NumNeighbors',i,'Distance','hamming');
    cvknn1 = crossval(k11);
    errR(i)= kfoldLoss(cvknn1);
end
errR

% Plot of Validation KfoldLoss vs k
plot([1:25],errR)
title('KNN CV loss');
xlabel('k');
ylabel('CV Error');

% Minimum kFoldloss  value from cross validation
errR(1)=1;
minimum = min(errR)
% Index of the minimum k
I = find(errR==minimum)

%Thus we get the loss from k=3
%Best k
bknn1 = fitcknn([trainV.P; valV.P],[trainV.T; valV.T],'CategoricalPredictors','all','NumNeighbors',3,'Distance','hamming');

%resubloss loss for best test
resub_best=resubLoss(bknn1)

%Error for optimal model
Y_k2= predict(bknn1,testV.P)
errR_k2= sum(testV.T~= Y_k2)/length(testV.T)

%Obtaining the confusion matrix
confusionmat(testV.T,Y_k2)