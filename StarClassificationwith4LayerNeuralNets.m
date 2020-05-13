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

% split train and test data

randrow=randperm(size(data, 1));
m=round(size(data,1)*0.8); %train sample size m
data_train=data_clean(randrow(1:m),:);
data_test=data_clean(randrow(m+1:end),:);
X_train=data_train(:,1:end-1);
y_train=data_train(:,end);
X_test=data_test(:,1:end-1);
y_test=data_test(:,end);




for i=1:10  % Ceteris paribus, see how train accuracy and test accuracy react to hidden neurons in each hidden layer.

% construct 4-layer neural networks and train 
m=round(size(data,1)*0.8); %train sample size m

L1=20; %input layer neurons
L2=i*100; %hidden layer neurons
L3=i*100; %hidden layer neurons
L4=6;%output layer neurons

rng(180);

theta1=2.*rand(L2,L1+1)-1; theta2=2.*rand(L3,L2+1)-1; theta3=2.*rand(L4,L3+1)-1;


for mm=1:m
    for label=1:L4
        X4_real(mm,label)=(label==y_train(mm,1));
    end
end

initial_nn_params = [theta1(:) ; theta2(:); theta3(:)];

lambda = 1;


%fprintf('Cost at parameters: %f',  nnCostFunction(initial_nn_params, L1, L2, L3, L4, X_train, y_train, lambda)); 


options = optimset('MaxIter', 500);
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, L1, L2, L3, L4, X_train, y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the neural network parameters)
[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);

theta1 = reshape( nn_params(1:L2 * (L1 + 1)), L2, (L1 + 1)    );
theta2 = reshape( nn_params( L2 * (L1 + 1)+1 : L2 * (L1 + 1)+ L3*(L2+1)), L3, L2+1 );
theta3 = reshape( nn_params( L2 * (L1 + 1)+L3*(L2+1)+1 : end), L4, L3+1 );

X1=[ones(m,1) X_train];
X2=[ones(m,1) sigmoid(X1*theta1') ];
X3=[ones(m,1) sigmoid(X2*theta2') ];
X4=sigmoid(X3*theta3');
[~,y_pred]=max(X4,[],2);
train_accuracy(i,1)=sum(y_pred==y_train)/m;
%fprintf('Cost at parameters: %f\n',  nnCostFunction(nn_params, L1, L2, L3, L4, X_train, y_train, lambda));
%fprintf('Training accuracy: %f\n',  train_accuracy);


% test

m =  size(data,1) - m ; % test sample size
X1=[ones(m,1) X_test];
X2=[ones(m,1) sigmoid(X1*theta1') ];
X3=[ones(m,1) sigmoid(X2*theta2') ];
X4=sigmoid(X3*theta3');
[~,y_pred]=max(X4,[],2);
test_accuracy(i,1)=sum(y_pred==y_test)/m;
%fprintf('Cost at parameters: %f\n',  nnCostFunction(nn_params, L1, L2, L3, L4, X_test, y_test, lambda));
%fprintf('Test accuracy: %f\n',  test_accuracy);

i
end

%plot how train accuracy and test accuracy react to hidden neurons in each hidden layer 
figure; hold on;
plot(100:100:1000,train_accuracy);
plot(100:100:1000,test_accuracy);
legend("train accuracy", "test accuracy","Location","west");
xlabel('hidden neurons in each hidden layer');
ylabel("accuracy");
hold off;
