restoredefaultpath;
clc;
clear all;
root = 'D:\PRM_HOSS\Matlab_code_vector_var';
output_dir = 'Perky_effect';
addpath(fullfile(root))
addpath(fullfile(root,'Utilities'))
addpath(fullfile(root,output_dir))
cd(root)

%% Model parameters
Aprior  = 0.9; % imagery vividness
App     = 2000;

Rprior  = 0.1; % assume low precision
Rpp     = 100:100:2000;
nRpp    = length(Rpp);

Wprior  = 0.9; % imagine W1
Wlambda = 50;

gen_mu     = [2 0];

gen_lambda = 3; % high precision input
nLam       = length(gen_lambda);

%% Generate data and perform inference

nSamples    = 100;
nReps       = 10;

p_a         = zeros(nRpp,nLam,nReps);
p_r         = zeros(nRpp,nLam,nReps);
p_w         = zeros(3,nRpp,nLam,nReps);

for r = 1:nRpp

    fprintf('Prior precision %d \n',r)

    for l = 1:nLam

        fprintf('\t Input precision %d \n',l)

        for i = 1:nReps

            gen_sigma = [1./gen_lambda(l) 0; 0 1./gen_lambda(l)];
            R = chol(gen_sigma);

            X = repmat(gen_mu,nSamples,1) + randn(nSamples,2)*R;

            % evaluate model
            samples = HOSS_evaluation_precision(X, Aprior, Rprior, Wprior, nSamples, Wlambda, App, Rpp(r));

            % Extract maximum a posteriori (MAP) estimates
            [f,xi] = ksdensity(samples.pA(:));
            [~,idx] = max(f);
            p_a(r,l,i) = xi(idx); % presence

            [f,xi] = ksdensity(samples.pR(:));
            [~,idx] = max(f);
            p_r(r,l,i) =  xi(idx); % reality

            for w0 = 1:3
                tmp = samples.pW(:,:,w0);
                [f,xi] = ksdensity(tmp(:));
                [~,idx] = max(f);
                p_w(w0,r,l,i) = xi(idx);
            end
        end
    end
end


save(fullfile(root,output_dir,'simulation_PerkyEffect_belief.mat'))
load(fullfile(root,output_dir,'simulation_PerkyEffect_belief.mat'))

%% Plot the results
% relative w1/w2
p_w1 = squeeze(p_w(2,:,:,:)./(p_w(2,:,:,:)+p_w(3,:,:,:)));

figure; 
subplot(1,2,1)
cW_map = makeColorMaps('teals');
cA_map = makeColorMaps('maroon');

a_idx = [100]; r_idx = [200];
w_idx = [150];

% a state
hold on; plotCI(squeeze(p_a(:,1,:)),Rpp,'CI',cA_map(a_idx,:),cA_map(a_idx,:),'over');
hold on; plot(Rpp,squeeze(mean(p_a(:,1,:),3)),'Color',cA_map(a_idx,:),'LineWidth',2)

% w state
hold on; plotCI(squeeze(p_w1(:,1,:)),Rpp,'CI',cW_map(w_idx,:),cW_map(w_idx,:),'over');
hold on; plot(Rpp,mean(p_w1(:,1,:),3),'Color',cW_map(w_idx,:),'LineWidth',2)

% r state
hold on; plotCI(squeeze(p_r(:,1,:)),Rpp,'CI',cA_map(r_idx,:),cA_map(r_idx,:),'over');
hold on; plot(Rpp,squeeze(mean(p_r(:,1,:),3)),'Color',cA_map(r_idx,:),'LineWidth',2)

hold on; ylim([0 1]); plot(xlim,[0.5 0.5],'k--');

subplot(1,2,2); idx = [1 20];
inferences = cat(3,p_w1(idx,:),squeeze(p_a(idx,1,:)),squeeze(p_r(idx,1,:)));
barwitherr(squeeze(std(inferences,[],2))./sqrt(nReps),squeeze(mean(inferences,2)));
set(gca,'XTickLabel',{'No Perky','Perky'});
