function X = HOSSPRM_sample(Aprior,App,Rprior,Rpp,w1prior,perceptlambda,nsamples)

% Define model structure
mu = nan(3, 2);
mu(1,:) = [0, 0];
mu(2,:) = [2, 0];
mu(3,:) = [0, 2];

% Sample A/R
betaA = estBetaParameters(Aprior,1/App);
betaR = estBetaParameters(Rprior,1/Rpp);
Aprior = betarnd(betaA(1), betaA(2));
Rprior = betarnd(betaR(1), betaR(2));

% Dirichlet distribution
w2prior = 1 - w1prior;
alpha = zeros(1, 3);
alpha(1) = perceptlambda * (1 - Aprior);
alpha(2) = perceptlambda * Aprior * w1prior;
alpha(3) = perceptlambda * Aprior * w2prior;

% Sample from Dirichlet prior over W
sampleW = zeros(1, length(alpha));
pW = zeros(1, length(alpha));
for k = 1:length(alpha)
    sampleW(k) = gamrnd(alpha(k), 1);  % MATLAB's gamma distribution
end
pW(1:3) = sampleW / sum(sampleW);

% Prior on sensory precision
mu_gamma = 1 + 2 * Rprior;
sigma_gamma = 0.1;
alpha_gamma = mu_gamma^2 / sigma_gamma^2;
beta_gamma = mu_gamma / sigma_gamma^2;
senselambda = gamrnd(alpha_gamma, 1 / beta_gamma);

% Covariance matrix for sensory precision
Sigma = [1/senselambda, 0; 0, 1/senselambda];

% Draw samples of X from mixture of Gaussians conditional on W
Z = randsample(1:length(pW), nsamples, true, pW);  % Random samples from pW
X = zeros(nsamples, 2);

for i = 1:nsamples
    if Z(i) == 1
        X(i,:) = mvnrnd(mu(1,:), Sigma);  % Multivariate normal distribution in MATLAB
    elseif Z(i) == 2
        X(i,:) = mvnrnd(mu(2,:), Sigma);
    else
        X(i,:) = mvnrnd(mu(3,:), Sigma);
    end
end

end