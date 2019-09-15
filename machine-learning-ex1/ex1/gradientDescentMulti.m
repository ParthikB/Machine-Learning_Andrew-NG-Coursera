function [theta,J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
s = size(theta)(1,1);
for iter = 1:num_iters

		for j = 1:s
			theta(j,1) = theta(j,1) - (alpha*( sum(((X*theta)-y).*X(:,j)) ))/m;
		end	
	
	
	J_history(iter) = computeCostMulti(X, y, theta)
	
	% hold on;
	% plot(iter, theta(1,1), 'r.', 'MarkerSize', 2);
	% xlabel('No. of iterations');
	% ylabel('Theta');
	% %plot((iter+10000),num_iters,'rx', 'MarkerSize',10);
	% plot(iter, theta(2,1), 'b.', 'MarkerSize', 2);
	% plot(iter, theta(3,1), 'g.', 'MarkerSize', 2);
	
end

% hold off;

end
