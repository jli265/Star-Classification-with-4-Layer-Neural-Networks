clear;clc;close;
data=readtable("Star Data.csv");

% clean data and do one hot encoding for star color, spectral class and star type

starcolor_data=data{:,6};
starcolor=zeros(size(data,1),9);
starcolor(:,1)= lower(strtrim(starcolor_data))=="red";
starcolor(:,2)= lower(strtrim(starcolor_data))=="blue";
starcolor(:,3)= (lower(strtrim(starcolor_data))== "white") + (lower(strtrim(starcolor_data)) == "whitish")  ;
starcolor(:,4)= lower(strtrim(starcolor_data))== "yellowish"   ;
starcolor(:,5)= (lower(strtrim(starcolor_data))== "yellowish white") + (lower(strtrim(starcolor_data)) == "white-yellow")...
    + (lower(strtrim(starcolor_data)) == "yellow-white")  ;
starcolor(:,6)= (lower(strtrim(starcolor_data))== "blue white") + (lower( strtrim(starcolor_data)) == "blue-white"); 
starcolor(:,7)= lower(strtrim(starcolor_data))== "orange"; 
starcolor(:,8)= lower(strtrim(starcolor_data))== "orange-red"; 
starcolor(:,9)= lower(strtrim(starcolor_data))== "pale yellow orange";

spectralclass_data=data{:,7};
spectralclass=zeros(size(data,1),7);
spectralclass(:,1)= lower(strtrim(spectralclass_data))=="a";
spectralclass(:,2)= lower(strtrim(spectralclass_data))=="b";
spectralclass(:,3)= lower(strtrim(spectralclass_data))=="f";
spectralclass(:,4)= lower(strtrim(spectralclass_data))=="g";
spectralclass(:,5)= lower(strtrim(spectralclass_data))=="k";
spectralclass(:,6)= lower(strtrim(spectralclass_data))=="m";
spectralclass(:,7)= lower(strtrim(spectralclass_data))=="o";

startype=data{:,5};
startype(startype==0)=6;


data_clean=[(data{:,1:4} -mean(data{:,1:4}))./std(data{:,1:4})  ,starcolor,spectralclass,startype]; % Feature Scaling
X=data_clean(:,1:end-1);
y=data_clean(:,end);

% 6-2-2 split train, validation and test data

m = size(X, 1); % m = Number of examples for training, validation and test

randrow=randperm(size(X, 1));
m_train=round(size(X,1)*0.6);
m_validation=round(size(X,1)*0.2);
m_test=m-m_train-m_validation;
X_train=X(randrow(1:m_train),:);
y_train=y(randrow(1:m_train),:);
X_validation=X(randrow(m_train+1:m_train+m_validation),:);
y_validation=y(randrow(m_train+1:m_train+m_validation),:);
X_test=X(randrow(m_train+m_validation+1 : end),:);
y_test=y(randrow(m_train+m_validation+1 : end),:);


% Neural Networks training and fine-tuning

% train data
lambda_all = [0, 0.01, 0.03,0.1,0.3,1,3];
cv_accuracy_init=0; % set initial value of cross validation accuracy before search for the best model parameter
lambda_best=0.01; % set initial value of best lambda
hid_layer_best=400; % set initial value of each hidden layer's neurons
count=0;
for i=1:3 % Ceteris paribus, see how train accuracy and validation accuracy react to hidden neurons in each hidden layer.
 for j=1:length(lambda_all)  %Ceteris paribus, see how train accuracy and validation accuracy react to lambda.
 lambda=lambda_all(j); 
 hid_layer=i*200+200;
 
% construct 4-layer neural networks and train 


L1=20; %input layer neurons
L2=hid_layer; %hidden layer neurons
L3=hid_layer; %hidden layer neurons
L4=6;%output layer neurons

rng(180);

theta1=2.*rand(L2,L1+1)-1; theta2=2.*rand(L3,L2+1)-1; theta3=2.*rand(L4,L3+1)-1;


% for mm=1:m
%     for label=1:L4
%         X4_real(mm,label)=(label==y_train(mm,1));
%     end
% end

initial_nn_params = [theta1(:) ; theta2(:); theta3(:)];




%fprintf('Cost at parameters: %f',  nnCostFunction(initial_nn_params, L1, L2, L3, L4, X_train, y_train, lambda)); 


options = optimset('MaxIter', 1000);
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, L1, L2, L3, L4, X_train, y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the neural network parameters)
[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);

theta1 = reshape( nn_params(1:L2 * (L1 + 1)), L2, (L1 + 1)    );
theta2 = reshape( nn_params( L2 * (L1 + 1)+1 : L2 * (L1 + 1)+ L3*(L2+1)), L3, L2+1 );
theta3 = reshape( nn_params( L2 * (L1 + 1)+L3*(L2+1)+1 : end), L4, L3+1 );

X1_validation=[ones(m_train,1) X_train];
X2=[ones(m_train,1) sigmoid(X1_validation*theta1') ];
X3=[ones(m_train,1) sigmoid(X2*theta2') ];
X4=sigmoid(X3*theta3');
[~,y_pred_train]=max(X4,[],2);
train_accuracy(i,j)=mean(y_pred_train==y_train);
%fprintf('Cost at parameters: %f\n',  nnCostFunction(nn_params, L1, L2, L3, L4, X_train, y_train, lambda));
%fprintf('Training accuracy: %f\n',  train_accuracy);


% cross validate data

X1_validation=[ones(m_validation,1) X_validation];
X2=[ones(m_validation,1) sigmoid(X1_validation*theta1') ];
X3=[ones(m_validation,1) sigmoid(X2*theta2') ];
X4=sigmoid(X3*theta3');
[~,y_pred_validation]=max(X4,[],2);
validation_accuracy(i,j)=mean(y_pred_validation==y_validation);
%fprintf('Cost at parameters: %f\n',  nnCostFunction(nn_params, L1, L2, L3, L4, X_test, y_test, lambda));
%fprintf('Test accuracy: %f\n',  test_accuracy);
count=count+1;
fprintf('Iterations: %d\n',  count);

% find optimal model from maximum cross validation accuracy
if validation_accuracy(i,j) > cv_accuracy_init
  cv_accuracy_best = validation_accuracy(i,j);
  lambda_best=lambda;
  hid_layer_best=hid_layer;
  theta1_best=theta1;
  theta2_best=theta2;
  theta3_best=theta3;
  nn_params_best=[theta1_best(:) ; theta2_best(:); theta3_best(:)];
  cv_accuracy_init=validation_accuracy(i,j);

end
 
 end

end

%plot how train accuracy and test accuracy react to hidden neurons in each hidden layer 
close; figure(1); hold on;
plot(lambda_all, train_accuracy(1,:)) ; plot(lambda_all, train_accuracy(2,:)) ; plot(lambda_all, train_accuracy(3,:)) ;
hold on;
plot(lambda_all, validation_accuracy(1,:)) ; plot(lambda_all, validation_accuracy(2,:)) ; plot(lambda_all, validation_accuracy(3,:)) ;
legend( "train accuracy (800 hidden neurons)", "train accuracy (1200 hidden neurons)",...
    "train accuracy (1600 hidden neurons)",...
    "validation accuracy (800 hidden neurons)", "validation accuracy (1200 hidden neurons)",...
    "validation accuracy (1600 hidden neurons)",...
    "Location","southwest");
xlabel("lambda");
ylabel("accuracy");
title("Model Selection");
hold off;



% calculate test accuracy
X1_test=[ones(m_test,1) X_test];
X2=[ones(m_test,1) sigmoid(X1_test*theta1_best') ];
X3=[ones(m_test,1) sigmoid(X2*theta2_best') ];
X4=sigmoid(X3*theta3_best');
[~,y_pred_test]=max(X4,[],2);
test_accuracy=mean(y_pred_test==y_test);

fprintf('Cost at parameters: %f\n',  nnCostFunction(nn_params_best, L1, hid_layer_best, hid_layer_best, L4, X_test, y_test, lambda_best));
fprintf('Test accuracy is %.3f.\n',  test_accuracy);
fprintf("Best lambda is %.3f.\n", lambda_best);
fprintf("Optimal number of hidden neurons is %d.\n", hid_layer_best*2);