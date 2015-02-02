function [ J, grad ] = costFunction(thetas, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

    Theta1 = reshape(thetas(1:hidden_layer_size * (input_layer_size + 1)), ...
        hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(thetas((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    	num_labels, (hidden_layer_size + 1));
    
    m = size(X, 1);
    J = 0;
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));
    
    a1 = [ones(size(X, 1), 1), X];
    z2 = Theta1 * a1';
    a2 = [ones(size(z2, 2), 1)'; sigmoid(z2)];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    yi = zeros(num_labels, m);
    for i=1:num_labels
        yi(i,:) = y==i;
    end
    A = -yi.*log(a3) - (1 - yi) .* log(1- a3);
    J = sum(sum(A))/m + (lambda/(2*m)) * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));
    d3 = a3 - yi;
    d2 = Theta2' * d3;
    d2 = d2(2:end, :) .* sigmoidGradient(z2);
    Theta1_grad = (d2 * a1)/m;
    Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda/m * Theta1(:, 2:end);
    Theta2_grad = (d3 * a2')/m;
    Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda/m * Theta2(:, 2:end);

    grad = [Theta1_grad(:) ; Theta2_grad(:)];
end

