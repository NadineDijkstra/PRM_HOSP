restoredefaultpath;
clc;
clear all;
root = 'D:\PRM_HOSP\Matlab_code_vector_var';
output_dir = 'Mental_imagery';
addpath(fullfile(root))
addpath(fullfile(root,'Utilities'))
addpath(fullfile(root,output_dir))
cd(root)

%% Model parameters
Aprior  = 0.9; 
prior_precisions = [100:100:2000];
nPP     = length(prior_precisions);

Wprior  = 0.9; % imagine W1 

Rprior  = 0.5; % flat precision prior 
Rpp     = 12; 

Wlambda = 50;


gen_lambda = 1; 
gen_mu     = [0 0];

%% Generate data and perform inference
nSamples    = 100;
nReps       = 10;

p_a         = zeros(1,nPP,nReps);
p_r         = zeros(1,nPP,nReps);
p_w         = zeros(3,nPP,nReps);

% loop over parameter values
for a = 1:nPP
    App = prior_precisions(a);

    fprintf('Aprior %d  \n',a)

    for i = 1:nReps

        gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
        R = chol(gen_sigma);

        X = repmat(gen_mu,nSamples,1) + randn(nSamples,2)*R;

        % evaluate model
        samples = HOSS_evaluation_precision(X, Aprior, Rprior, Wprior, nSamples, Wlambda, App, Rpp);
     

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

save(fullfile(root,output_dir,'simulation_mentalimagery.mat'))
load(fullfile(root,output_dir,'simulation_mentalimagery.mat'))

%% Visualization
figure; 
subplot(1,2,1);
cW_map = makeColorMaps('teals');
cA_map = makeColorMaps('maroon');

plotCI(squeeze(p_a),prior_precisions,'CI',cA_map(100,:),cA_map(100,:),'over');
hold on; plot(prior_precisions,squeeze(mean(p_a,3)),'Color',cA_map(100,:),'LineWidth',2)
hold on; plotCI(p_w1,prior_precisions,'CI',cW_map(150,:),cW_map(150,:),'over');
hold on; plot(prior_precisions,mean(p_w1,2),'Color',cW_map(150,:),'LineWidth',2)

hold on; ylim([0 1]); plot(xlim,[0.5 0.5],'k--');

subplot(1,2,2);
idx = [1 20]; % extreme ends of vividness
inferences = cat(3,squeeze(p_w1(idx,:)),squeeze(p_a(:,idx,:)),squeeze(p_r(:,idx,:)));
barwitherr(squeeze(std(inferences,[],2))./sqrt(nReps),squeeze(mean(inferences,2)));
set(gca,'XTickLabel',{'Aphantasia','Imagery'});
