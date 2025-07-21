function samples = HOSS_evaluation_precision(X, Aprior, Rprior, Wprior, nsamples, Wlambda, App, Rpp)

% Define model structure
mu(1,:) = [0 0]; % w0
mu(2,:) = [2 0]; % w1
mu(3,:) = [0 2]; % w2

% Uninformative priors
if nargin < 7
    App = 12; Rpp = 12; % precision of the priors (12 is flat/[1 1])
end

% Convert priors to beta Parameters
betaA = estBetaParameters(Aprior,1/App);
betaR = estBetaParameters(Rprior,1/Rpp);

% Run chain for experimental data
parameters = {'pA', 'pW', 'pR','senselambda'};  % The parameter(s) to be monitored.
adaptSteps = 1000;        % Number of steps to "tune" the samplers.
nBurnin = 1000;           % Number of steps to "burn-in" the samplers.
nChains = 3;              % Number of chains to run.
numSavedSteps=5000;       % Total number of steps in chains to save.
thinSteps=1;              % Number of steps to "thin" (1=keep every step).
nIter = ceil( ( numSavedSteps * thinSteps ) / nChains ); % Steps per chain.

w1prior = Wprior;
w2prior = 1-Wprior;
data = struct('X',X,'w1prior',w1prior,'w2prior',w2prior,...
'aA',betaA(1),'bA',betaA(2),'aR',betaR(1),'bR',betaR(2),...
'mu',mu,'nsamples',nsamples,'perceptlambda',Wlambda); % with precision on the A and R priors
% 
% data = struct('X',X,'w1prior',w1prior,'w2prior',w2prior,...
% 'mu',mu,'nsamples',nsamples,'perceptlambda',50);

% initial values latent variables
for c = 1:nChains
    init0(c) = struct;
end

samples = matjags(data,...
    fullfile(pwd, 'HOSS_precision_model_R.txt'), init0 ,...
    'nChains', nChains, 'monitorParams', parameters, ...
    'nBurnin', nBurnin, 'nSamples', nIter, ...
    'verbosity',0);
