%% Simulate the imagery binocular rivalry priming effect
restoredefaultpath;
clc;
clear all;
root = 'D:\PRM_HOSP\Matlab_code_vector_var';
output_dir = 'EmpiricalSignatures
';
addpath(fullfile(root))
addpath(fullfile(root,'Utilities'))
addpath(fullfile(root,output_dir))
cd(root)

%% Simulation settings
Aprior      = [0.5 0.9 0.9]; % no imagery - imagery
App         = [12 100 100]; % vividness
Wprior      = [0.5 0.9 0.1]; % no imagery - imagery congruent - imagery incongruent
Wlambdas    = [5 30 30];

nPP     = length(App);

Rprior  = 0.5; % flat precision prior 
Rpp     = 12; 

gen_lambda = 3; % high precision input
gen_mu     = [1.4 1]; % slightly biased content input to induce perceptual stability

%% Generate data and perform inference
nSamples    = 100;
nReps       = 10;

p_a         = zeros(nPP,nReps);
p_r         = zeros(nPP,nReps);
p_w         = zeros(3,nPP,nReps);

% loop over parameter values
for a = 1:nPP

    fprintf('Aprior %d  \n',a)

    for i = 1:nReps

        gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
        R = chol(gen_sigma);

        X = repmat(gen_mu,nSamples,1) + randn(nSamples,2)*R;

        % evaluate model
        samples = HOSS_evaluation_precision(X, Aprior(a), Rprior, Wprior(a), nSamples, Wlambdas(a), App(a), Rpp);

        % Extract maximum a posteriori (MAP) estimates
        [f,xi] = ksdensity(samples.pA(:));
        [~,idx] = max(f);
        p_a(1,a,i) = xi(idx); % presence

        [f,xi] = ksdensity(samples.pR(:));
        [~,idx] = max(f);
        p_r(1,a,i) =  xi(idx); % reality

        for w0 = 1:3
            tmp = samples.pW(:,:,w0);
            [f,xi] = ksdensity(tmp(:));
            [~,idx] = max(f);
            p_w(w0,a,i) = xi(idx);
        end
    end
end

save(fullfile(root,output_dir,'simulation_binocularrivalry.mat'))
load(fullfile(root,output_dir,'simulation_binocularrivalry.mat'))

%% Plot results
% perceptual dominance
p_w1 = squeeze(p_w(2,:,:))./squeeze(p_w(2,:,:)+p_w(3,:,:)); % how much more likely is w1 over w2
figure;
barwitherr(std(p_w1,[],2)./sqrt(nReps),mean(p_w1,2));
set(gca,'XTickLabels',{'No imagery','Congruent','Incongruent'});
ylim([0.4 1]); hold on; plot(xlim,[0.5 0.5],'k--')

%% Influence of prior precision on BR priming 
Aprior      = 0.9; % no imagery - imagery
App         = [100:500:2600]; % vividness
Wprior      = 0.9; % no imagery - imagery congruent - imagery incongruent
Wlambdas    = 50;

nPP     = length(App);

Rprior  = 0.5; % flat precision prior 
Rpp     = 12; 

gen_lambda = 3; % high precision input
gen_mu     = [1 1]; % ambiguous

nSamples    = 100;
nReps       = 10;

p_a         = zeros(nPP,nReps);
p_r         = zeros(nPP,nReps);
p_w         = zeros(3,nPP,nReps);

% loop over parameter values
for a = 1:nPP

    fprintf('Aprior %d  \n',a)

    for i = 1:nReps

        gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
        R = chol(gen_sigma);

        X = repmat(gen_mu,nSamples,1) + randn(nSamples,2)*R;

        % evaluate model
        samples = HOSS_evaluation_precision(X, Aprior, Rprior, Wprior, nSamples, Wlambdas, App(a), Rpp);

        % Extract maximum a posteriori (MAP) estimates
        [f,xi] = ksdensity(samples.pA(:));
        [~,idx] = max(f);
        p_a(1,a,i) = xi(idx); % presence

        [f,xi] = ksdensity(samples.pR(:));
        [~,idx] = max(f);
        p_r(1,a,i) =  xi(idx); % reality

        for w0 = 1:3
            tmp = samples.pW(:,:,w0);
            [f,xi] = ksdensity(tmp(:));
            [~,idx] = max(f);
            p_w(w0,a,i) = xi(idx);
        end
    end
end

save(fullfile(root,output_dir,'simulation_binocularrivalry_vividness.mat'))
load(fullfile(root,output_dir,'simulation_binocularrivalry_vividness.mat'))

%% Plot the results
p_w1 = squeeze(p_w(2,:,:))./squeeze(p_w(2,:,:)+p_w(3,:,:)); % how much more likely is w1 over w2

figure;
barwitherr(std(p_w1,[],2)./sqrt(nReps),mean(p_w1,2));
hold on; plot(xlim,[0.5 0.5],'k--')
xlabel('Imagery vividness'); ylabel('Dominance');
ylim([0.4 1]); 

% test significance 
Y = reshape(p_w1,nPP*nReps,1);
X = [ones(nPP*nReps,1) reshape(repmat(1:nPP,nReps,1),nPP*nReps,1)];
[b,~,~,~,stats] = regress(Y,X);
