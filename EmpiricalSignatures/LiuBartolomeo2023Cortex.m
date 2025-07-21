restoredefaultpath;
clc;
clear all;
root = 'D:\PRM_HOSP\Matlab_code_vector_var';
output_dir = 'Expectation';
addpath(fullfile(root))
addpath(fullfile(root,'Utilities'))
addpath(fullfile(root,output_dir))
cd(root)

%% Settings
Wlambdas     = [50 50]; % precision of W priors

gen_mu(:,1) = 0:0.2:2; % varying input strength
gen_mu(:,2) = zeros(length(gen_mu),1);

gen_lambda  = [2 3]; % varying input precision

Wprior      = 0.9; % perceptual expectations

Rprior      = 0.5; % flat
Rpp         = 12;

Aprior      = 0.5; % flat
App         = 12;

nReps = 10;

%% Generate data and perform inference
nPrecision  = length(gen_lambda);
nMu         = length(gen_mu);
nPW         = length(Wlambdas)+1;

nSamples    = 100; % See if this still works

p_a         = zeros(nMu,nPrecision,nPW,nReps);
p_r         = zeros(nMu,nPrecision,nPW,nReps);
p_w         = zeros(3,nMu,nPrecision,nPW,nReps);

% loop over the parameter values
allSamples = cell(nMu,nPrecision,nPW);
for mu = 1:nMu
    fprintf('Mu %d out of %d \n',mu,nMu)
    for p = 1:nPrecision
        fprintf('\t Precision: %d',p)
        for pw = 1:nPW
            fprintf('\t W prior: %d \n',pw)

            for r = 1:nReps
                gen_sigma = [1./gen_lambda(p) 0; 0 1./gen_lambda(p)];
                R = chol(gen_sigma);

                X = repmat(gen_mu(mu,:),nSamples,1) + randn(nSamples,2)*R;

                % evaluate model
                if pw == 1
                    samples = HOSS_evaluation_precision(X, Aprior, Rprior, 0.5, nSamples, 5, App, Rpp);
                else
                    samples = HOSS_evaluation_precision(X, Aprior, Rprior, Wprior, nSamples, Wlambdas(pw-1), App, Rpp);
                end
                allSamples{mu,p,pw} = samples;              

                % Extract maximum a posteriori (MAP) estimates
                [f,xi] = ksdensity(samples.pA(:));
                [~,idx] = max(f);
                p_a(mu,p,pw,r) = xi(idx); % presence

                [f,xi] = ksdensity(samples.pR(:));
                [~,idx] = max(f);
                p_r(mu,p,pw,r) =  xi(idx); % reality

                for w0 = 1:3                    
                    tmp = samples.pW(:,:,w0);
                    [f,xi] = ksdensity(tmp(:));
                    [~,idx] = max(f);
                    p_w(w0,mu,p,pw,r) = xi(idx);
                end
            end
        end
    end
end

save(fullfile(root,output_dir,'simulation_expectation.mat'))
load(fullfile(root,output_dir,'simulation_expectation.mat'))

%% Visualisation
p_wALL = p_w; p_aALL  = p_a; p_rALL = p_r;
p_w = mean(p_wALL,5);
p_a = mean(p_aALL,4);
p_r = mean(p_rALL,4);

figure; % relative w1/w2
subplot(1,2,1);
p = 2; % only high precision input
for pw = 1:2 % only no and weak prediction
    tmp = nan(nMu,nReps);
    for mu = 1:nMu
        t = p_wALL(2,mu,p,pw,:)./(p_wALL(2,mu,p,pw,:)+p_wALL(3,mu,p,pw,:));
        tmp(mu,:) = t(:); clear t
    end
    plotCI(tmp,gen_mu(:,1)','CI',csW(pw,:),csW(pw,:),'over'); hold on;
    plot(gen_mu(:,1),mean(tmp,2),'Color',csW(pw,:), 'LineWidth',2); hold on;
end
legend('No prediction','Predict W1');
ylabel('Posterior W1'); ylim([0 1])

hold on;
for pw = 1:2 % only no and weak prediction
    tmp = nan(nMu,nReps);
    for mu = 1:nMu
        t = p_aALL(mu,p,pw,:);
        tmp(mu,:) = t(:); clear t
    end
    plotCI(tmp,gen_mu(:,1)','CI',csA(pw,:),csA(pw,:),'over'); hold on;
    plot(gen_mu(:,1),mean(tmp,2),'Color',csA(pw,:), 'LineWidth',2); hold on;
end

subplot(1,2,2); 
p_w1 = squeeze(mean(squeeze(p_wALL(2,:,p,1:2,:)./(p_wALL(2,:,p,1:2,:)+p_wALL(3,:,p,1:2,:))),1));
inferences = cat(3,p_w1,squeeze(mean(p_aALL(:,p,1:2,:),1)),squeeze(mean(p_rALL(:,p,1:2,:),1)));
barwitherr(squeeze(std(inferences,[],2))./sqrt(nReps),squeeze(mean(inferences,2)));
set(gca,'XTickLabel',{'No prediction','Prediction'});
