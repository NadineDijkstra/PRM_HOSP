function [X,Wprior_samples] = HOSSPRM_sample_firstorder_threestates(wprior,perceptlambda,senselambda,nsamples)

% Define model structure
mu = nan(3, 2);
mu(1,:) = [0, 0];
mu(2,:) = [2, 0];
mu(3,:) = [0, 2];

% Dirichlet distribution
alpha = zeros(1, 3);
alpha(1) = perceptlambda * wprior(1);
alpha(2) = perceptlambda * wprior(2);
alpha(3) = perceptlambda * wprior(3);

% Sample from Dirichlet prior over W
Wprior_samples = zeros(nsamples, 3);
for i = 1:nsamples
    gamma_vals = gamrnd(alpha, 1);
    Wprior_samples(i, :) = gamma_vals / sum(gamma_vals);
end

% Covariance matrix for sensory precision
Sigma = [1/senselambda, 0; 0, 1/senselambda];

% Draw samples of X from mixture of Gaussians conditional on W
X = zeros(nsamples, 2);
for i = 1:nsamples
    Z = randsample(1:3, 1, true, Wprior_samples(i,:));  % Random samples from pW
    
    if Z == 1
        X(i,:) = mvnrnd(mu(1,:), Sigma);  % Multivariate normal distribution in MATLAB
    elseif Z == 2
        X(i,:) = mvnrnd(mu(2,:), Sigma);
    elseif Z ==3
        X(i,:) = mvnrnd(mu(3,:), Sigma);
    end
end

end