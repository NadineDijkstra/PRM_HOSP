function samples = HOSSPRM_sample_from_prior(Aprior,App,Rprior,Rpp,wprior,perceptlambda,nSamples)

% beta parameters
betaA = estBetaParameters(Aprior,1/App);
betaR = estBetaParameters(Rprior,1/Rpp);

% sample from the top levels 
Aprior_samples = betarnd(betaA(1), betaA(2), nSamples, 1);
Rprior_samples = betarnd(betaR(1), betaR(2), nSamples, 1);

% Compute Dirichlet alpha parameters for each sample
alpha_samples = [perceptlambda * (1 - Aprior_samples), ...
                 perceptlambda * Aprior_samples .* wprior, ...
                 perceptlambda * Aprior_samples .* (1-wprior)];

% Sample from Dirichlet for low levels
Wprior_samples = zeros(nSamples, 3);
for i = 1:nSamples
    gamma_vals = gamrnd(alpha_samples(i, :), 1);
    Wprior_samples(i, :) = gamma_vals / sum(gamma_vals);
end

% Save
samples.pA = Aprior_samples;
samples.pR = Rprior_samples;
samples.pW = Wprior_samples;