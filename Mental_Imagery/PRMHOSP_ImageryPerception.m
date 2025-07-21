restoredefaultpath;
clc;
clear all;
root = 'D:\PRM_HOSS\Matlab_code_vector_var';
output_dir = 'Mental_imagery';
addpath(fullfile(root))
addpath(fullfile(root,'Utilities'))
addpath(fullfile(root,output_dir))
cd(root)

%% Settings
gen_mu(:,1) = 0:0.2:2; % varying input strength 
gen_mu(:,2) = zeros(length(gen_mu),1);

gen_lambda  = 3; % high input precision

Aprior      = [0.5 0.9];% 0.9]; % no imagery - imagery
App         = [12 100];% 100];
Wprior      = [0.5 0.9];% 0.1]; % no, congruent, incongruent 
Wlambdas    = [5 50];% 5]; % precision of W priors

nCond       = 2;

Rprior      = 0.5; % flat
Rpp         = 12;


nReps = 10;

%% Generate data and perform inference
nMu         = length(gen_mu);

nSamples    = 100;

p_a         = zeros(nMu,nCond,nReps);
p_r         = zeros(nMu,nCond,nReps);
p_w         = zeros(3,nMu,nCond,nReps);

% loop over the parameter values
allSamples = cell(nMu,nCond);
for mu = 1:nMu
    fprintf('Mu %d out of %d \n',mu,nMu)
        for c = 1:nCond
            fprintf('\t Cond: %d \n',c)

            for r = 1:nReps
                gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
                R = chol(gen_sigma);

                X = repmat(gen_mu(mu,:),nSamples,1) + randn(nSamples,2)*R;

                % evaluate model
                samples = HOSS_evaluation_precision(X, Aprior(c), Rprior, Wprior(c), nSamples, Wlambdas(c), App(c), Rpp);
                allSamples{mu,c} = samples;              

                % Extract maximum a posteriori (MAP) estimates
                [f,xi] = ksdensity(samples.pA(:));
                [~,idx] = max(f);
                p_a(mu,c,r) = xi(idx); % presence

                [f,xi] = ksdensity(samples.pR(:));
                [~,idx] = max(f);
                p_r(mu,c,r) =  xi(idx); % reality

                for w0 = 1:3                    
                    tmp = samples.pW(:,:,w0);
                    [f,xi] = ksdensity(tmp(:));
                    [~,idx] = max(f);
                    p_w(w0,mu,c,r) = xi(idx);
                end
            end
        end
end

save(fullfile(root,output_dir,'simulation_imageryperception.mat'))
load(fullfile(root,output_dir,'simulation_imageryperception.mat'))

%% Visualisation
p_wALL = p_w; p_aALL  = p_a; p_rALL = p_r;
p_w = mean(p_wALL,4);
p_a = mean(p_aALL,3);
p_r = mean(p_rALL,3);

c_map = makeColorMaps('teals');
csW   = c_map(round(linspace(60,256,nCond+1)),:);
c_map = makeColorMaps('maroon');
csA   = c_map(round(linspace(60,230,nCond+1)),:);


%% Paper figures

figure; % relative w1/w2
subplot(1,2,1);
for c = 1:nCond 
    tmp = nan(nMu,nReps);
    for mu = 1:nMu
        t = p_wALL(2,mu,c,:)./(p_wALL(2,mu,c,:)+p_wALL(3,mu,c,:));
        tmp(mu,:) = t(:); clear t
    end
    plotCI(tmp,gen_mu(:,1)','CI',csW(c,:),csW(c,:),'over'); hold on;
    plot(gen_mu(:,1),mean(tmp,2),'Color',csW(c,:), 'LineWidth',2); hold on;
end
legend('No imagery','Imagine W1');
ylabel('Posterior W1'); ylim([0 1])

hold on; % awareness inference
for c = 1:nCond 
    tmp = nan(nMu,nReps);
    for mu = 1:nMu
        t = p_aALL(mu,c,:);
        tmp(mu,:) = t(:); clear t
    end
    plotCI(tmp,gen_mu(:,1)','CI',csA(c,:),csA(c,:),'over'); hold on;
    plot(gen_mu(:,1),mean(tmp,2),'Color',csA(c,:), 'LineWidth',2); hold on;
end

subplot(1,2,2);
p_w1 = squeeze(mean(p_wALL(2,:,:,:)./(p_wALL(2,:,:,:)+p_wALL(3,:,:,:)),2));
inferences = cat(3,p_w1,squeeze(mean(p_aALL,1)),squeeze(mean(p_rALL,1)));
barwitherr(squeeze(std(inferences,[],2))./sqrt(nReps),squeeze(mean(inferences,2)));
set(gca,'XTickLabel',{'No imagery','Imagery'});
