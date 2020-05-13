function [J grad] = nnCostFunction(nn_params, ...
                                   L1, ...
                                   L2, ...
                                   L3, ...
                                   L4, ...
                                   X, y_real, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.


% Reshape nn_params back into the parameters 
theta1 = reshape(  nn_params(1:L2 * (L1 + 1)), L2, (L1 + 1)    );
theta2 = reshape( nn_params( L2 * (L1 + 1)+1 : L2 * (L1 + 1)+ L3*(L2+1)), L3, L2+1 );
theta3 = reshape( nn_params( L2 * (L1 + 1)+L3*(L2+1)+1 : end), L4, L3+1 );



% Setup some useful variables
m = size(X, 1); %sample size
         
% You need to return the following variables correctly 
J = 0;
theta1_grad = zeros(size(theta1));
theta2_grad = zeros(size(theta2));
theta3_grad = zeros(size(theta3));
% ====================== YOUR CODE HERE ======================
X1=[ones(m,1) X];
X2=[ones(m,1) sigmoid(X1*theta1') ];
X3=[ones(m,1) sigmoid(X2*theta2') ];
X4=sigmoid(X3*theta3');
[~,y_pred]=max(X4,[],2);


X4_real=zeros(m,L4);
for mm=1:m
    for label=1:L4
        X4_real(mm,label)=(label==y_real(mm,1));
    end
end
 
J1  = sum( sum( -X4_real.*log(X4)-(1-X4_real).*log(1-X4) )  )./m;
J2 = lambda/(2*m)*( sum(sum(theta1(:,2:end).^2)) +   sum(sum(theta2(:,2:end).^2))  + sum(sum(theta3(:,2:end).^2))  );
J=J1+J2;

 
delta4=(X4-X4_real)';
delta3=theta3'*delta4.*(X3'.*(1-X3')); delta3=delta3(2:end,:);
delta2=theta2'*delta3.*(X2'.*(1-X2')); delta2=delta2(2:end,:);


Delta3=delta4*X3./m;  Delta3(:,2:end)=Delta3(:,2:end)+lambda./m.*theta3(:,2:end);
Delta2=delta3*X2./m ; Delta2(:,2:end)=Delta2(:,2:end)+lambda./m.*theta2(:,2:end);
Delta1=delta2*X1./m ; Delta1(:,2:end)=Delta1(:,2:end)+lambda./m.*theta1(:,2:end);




theta3_grad=Delta3;
theta2_grad=Delta2;
theta1_grad=Delta1;


% Unroll gradients
grad = [theta1_grad(:) ; theta2_grad(:);theta3_grad(:)];



% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

 

end
