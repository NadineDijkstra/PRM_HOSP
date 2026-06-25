function samples = HOSPPRM_evaluation_firstorder(X, Wprior, Wlambda, senselambda, nsamples)

% Define model structure - only two states
mu(1,:) = [0 0]; % w0
mu(2,:) = [2 0]; % w1
mu(3,:) = [0 2]; % w2

% Run chain for experimental data
parameters = {'pW'};  % The parameter(s) to be monitored.
adaptSteps = 1000;        % Number of steps to "tune" the samplers.
nBurnin = 1000;           % Number of steps to "burn-in" the samplers.
nChains = 3;              % Number of chains to run.
numSavedSteps=5000;       % Total number of steps in chains to save.
thinSteps=1;              % Number of steps to "thin" (1=keep every step).
nIter = ceil( ( numSavedSteps * thinSteps ) / nChains ); % Steps per chain.

data = struct('X',X,'wprior0',Wprior(1),'wprior1',Wprior(2),'wprior2',Wprior(3),...
    'senselambda',senselambda,'mu',mu,...
    'nsamples',nsamples,'perceptlambda',Wlambda); % with precision on the A and R priors

% initial values latent variables
for c = 1:nChains
    init0(c) = struct;
end

samples = matjags(data,...
    fullfile(pwd, 'HOSPPRM_firstorder_model.txt'), init0 ,...
    'nChains', nChains, 'monitorParams', parameters, ...
    'nBurnin', nBurnin, 'nSamples', nIter, ...
    'verbosity',0);
