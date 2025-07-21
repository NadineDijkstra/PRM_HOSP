restoredefaultpath;
clc;
clear all;
root = 'D:\PRM_HOSP\Matlab_code_vector_var';
output_dir = 'Hallucination';
addpath(fullfile(root))
addpath(fullfile(root,'Utilities'))
addpath(fullfile(root,output_dir))
cd(root)

%% Model parameters
Aprior_u  = 0.9; 
Aprior_v = 2000; % A prior precision

Rprior_u  = 0.5; % increase prior on R
Rprior_v  = 12; 

Wprior  = 0.9; % content 
Wlambda = 50;

gen_lambda = 1:(1/3):3; % <--- spontaneous increase in precision
gen_mu     = [0 0];
nPP        = length(gen_lambda);


%% Generate data and perform inference
nSamples    = 100;
nReps       = 10;

p_a         = zeros(1,nPP,nReps);
p_r         = zeros(1,nPP,nReps);
p_w         = zeros(3,nPP,nReps);

% loop over parameter values
for a = 1:nPP

    fprintf('Precision %d  \n',a)

    for i = 1:nReps

        gen_sigma = [1./gen_lambda(a) 0; 0 1./gen_lambda(a)];
        R = chol(gen_sigma);

        X = repmat(gen_mu,nSamples,1) + randn(nSamples,2)*R;

        % evaluate model
        samples = HOSS_evaluation_precision(X, Aprior_u, Rprior_u, Wprior, nSamples, Wlambda, Aprior_v, Rprior_v);
     

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

save(fullfile(root,output_dir,'simulation_bottomup_hallucination.mat'))
load(fullfile(root,output_dir,'simulation_bottomup_hallucination.mat'))

%% Plot the results
% relative w1/w2
p_w1 = squeeze(p_w(2,:,:)./(p_w(2,:,:)+p_w(3,:,:)));

figure; 
subplot(1,2,1);
cW_map = makeColorMaps('teals');
cA_map = makeColorMaps('maroon');

plotCI(squeeze(p_a),gen_lambda,'CI',cA_map(100,:),cA_map(100,:),'over');
hold on; plot(gen_lambda,squeeze(mean(p_a,3)),'Color',cA_map(100,:),'LineWidth',2)
hold on; plotCI(p_w1,gen_lambda,'CI',cW_map(150,:),cW_map(150,:),'over');
hold on; plot(gen_lambda,mean(p_w1,2),'Color',cW_map(150,:),'LineWidth',2)

hold on; ylim([0 1]); plot(xlim,[0.5 0.5],'k--');

hold on; plotCI(squeeze(p_r),gen_lambda,'CI',cA_map(200,:),cA_map(200,:),'over');
hold on; plot(gen_lambda,squeeze(mean(p_r,3)),'Color',cA_map(200,:),'LineWidth',2)

subplot(1,2,2); idx = [2 6];
inferences = cat(3,p_w1(idx,:),squeeze(p_a(:,idx,:)),squeeze(p_r(:,idx,:)));
barwitherr(squeeze(std(inferences,[],2))./sqrt(nReps),squeeze(mean(inferences,2)));
set(gca,'XTickLabel',{'Low input precision','High input precision'});

