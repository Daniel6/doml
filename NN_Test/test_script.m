%% 1: Load Data and Thetas
lambda = 1;
input_layer_size = 400;
hidden_layer_size = 25;
num_labels = 10;

load('ex4weights.mat');
thetas = [Theta1(:); Theta2(:)];

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_Thetas = [initial_Theta1(:); initial_Theta2(:)];

load('ex4data1.mat');
m = size(X, 1);

%% 2: Train
options = optimset('MaxIter', 1000);

J = @(p) costFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

[thetas, cost] = fmincg(J, initial_Thetas, options);

Theta1 = reshape(thetas(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(thetas((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
%% 3: Predict

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);