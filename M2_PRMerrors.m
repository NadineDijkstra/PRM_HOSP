restoredefaultpath;
clc;
clear all;
root = [];
outDir = fullfile(root,'Model_comparison');
addpath(fullfile(fileparts(root),'Utilities'))
addpath(fullfile(outDir))
cd(outDir)


%% Increase prior on precision 
% priors
Wprior = [0.1/2 0.9 0.1/2];
Wlambda = 500; % strong priors for imagery 
Rprior = 0.9; 
Rpp    = 1000;

% simulation settings
nSamples    = 100;
nRep        = 10;

% input parameters
gen_lambda  = 1;
gen_mu      = [0 0];

% pre-allocate
allSamples  = cell(1,nRep);
p_w         = nan(3,nRep);
p_r           = nan(1,nRep);

% run simulations

for i = 1:nRep

    % generate data
    gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
    R = chol(gen_sigma);

    X = repmat(gen_mu,nSamples,1) + randn(nSamples,2)*R;

    % evaluate model
    samples = HOSS_evaluation_onlyprecision(X, Rprior, Wprior,nSamples, Wlambda, Rpp);
    allSamples{i} = samples;

    % Extract maximum a posteriori (MAP) estimates
    tmp = samples.pR(:);
    [f,xi] = ksdensity(tmp(:));
    [~,idx] = max(f);
    p_r(i) = xi(idx);

    for w0 = 1:3
        tmp = samples.pW(:,:,w0);
        [f,xi] = ksdensity(tmp(:));
        [~,idx] = max(f);
        p_w(w0,i) = xi(idx);
    end

end

save(fullfile(root,'Model_comparison','M2_TopDownHallucination.mat'))

%% Increase precision input w/ or w/o imagery
% priors
Wprior = [0.1/2 0.9 0.1/2];
Wlambda = 500; % weak vs strong priors for imagery
Rprior = 0.5;
Rpp    = 12;

% simulation settings
nSamples    = 100;
nRep        = 10;

% input parameters
gen_lambda  = 3;
gen_mu      = [0 0];

% pre-allocate
allSamples  = cell(1,nRep);
p_w         = nan(3,nRep);
p_r           = nan(1,nRep);

% run simulations

for i = 1:nRep

    % generate data
    gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
    R = chol(gen_sigma);

    X = repmat(gen_mu,nSamples,1) + randn(nSamples,2)*R;

    % evaluate model
    samples = HOSS_evaluation_onlyprecision(X, Rprior, Wprior,nSamples, Wlambda, Rpp);
    allSamples{i} = samples;

    % Extract maximum a posteriori (MAP) estimates
    tmp = samples.pR(:);
    [f,xi] = ksdensity(tmp(:));
    [~,idx] = max(f);
    p_r(i) = xi(idx);

    for w0 = 1:3
        tmp = samples.pW(:,:,w0);
        [f,xi] = ksdensity(tmp(:));
        [~,idx] = max(f);
        p_w(w0,i) = xi(idx);
    end

end

save(fullfile(root,'Model_comparison','M2_BottomUpHallucination.mat'))

%% Perky effect
% priors
Wprior = [0.1/2 0.9 0.1/2];
Wlambdas = 500; % weak vs strong imagery 
Rprior = 0.1; % strong prior on low precision
Rpp    = 1000;

% simulation settings
nSamples    = 100;
nRep        = 10;

% input parameters
gen_lambda  = 3; % high precision input
gen_mu      = [1 0]; % faint perception 

% pre-allocate
allSamples  = cell(1,nRep);
p_w         = nan(3,nRep);
p_r           = nan(1,nRep);

% run simulations
for w = 1:2

    fprintf('w %d of %d \n',w,length(Wlambdas))

    for i = 1:nRep

        % generate data
        gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
        R = chol(gen_sigma);

        X = repmat(gen_mu,nSamples,1) + randn(nSamples,2)*R;
   
        % evaluate model
        samples = HOSS_evaluation_onlyprecision(X, Rprior, Wprior,nSamples, Wlambdas(w), Rpp);
        allSamples{i} = samples;

        % Extract maximum a posteriori (MAP) estimates
        tmp = samples.pR(:);
        [f,xi] = ksdensity(tmp(:));
        [~,idx] = max(f);
        p_r(i) = xi(idx);

        for w0 = 1:3
            tmp = samples.pW(:,:,w0);
            [f,xi] = ksdensity(tmp(:));
            [~,idx] = max(f);
            p_w(w0,i) = xi(idx);
        end  

    end
end

save(fullfile(root,'Model_comparison','M2_PerkyEffect.mat'))


%% Plot the results
cW_map = makeColorMaps('teals');
cA_map = makeColorMaps('maroon');
figure;

% prior hallucination
load(fullfile(root,'Model_comparison','M2_TopDownHallucination.mat'),'p_w','p_r')

subplot(1,3,1)
inferences = cat(1,squeeze(p_w),[],squeeze(p_r));
barwitherr(squeeze(std(inferences,[],2))./sqrt(nRep),squeeze(mean(inferences,2)));
set(gca,'XTickLabels',{'w0','w1','w2','R'}); title('Increased R prior')


% bottom up hallucination
load(fullfile(root,'Model_comparison','M2_BottomUpHallucination.mat'),'p_w','p_r')
subplot(1,3,2)
inferences = cat(1,squeeze(p_w),[],squeeze(p_r));
barwitherr(squeeze(std(inferences,[],2))./sqrt(nRep),squeeze(mean(inferences,2)));
%ylim([0 1]);
set(gca,'XTickLabels',{'w0','w1','w2','R'}); title('Increased precision input')

% Perky effect
figure;
load(fullfile(root,'Model_comparison','M2_PerkyEffect.mat'),'p_w','p_r')
subplot(1,3,3)
inferences = cat(1,squeeze(p_w),[],squeeze(p_r));
barwitherr(squeeze(std(inferences,[],2))./sqrt(nRep),squeeze(mean(inferences,2)));
%ylim([0 1]);
set(gca,'XTickLabels',{'w0','w1','w2','R'}); tile('Perky effect')

