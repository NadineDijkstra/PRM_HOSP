restoredefaultpath;
clc;
clear all;
root = 'D:\PRM_HOSS\Matlab_code_vector_var';
addpath(fullfile(root))
addpath(fullfile(root,'Utilities'))
addpath(fullfile(root,'Perception'))
cd(root)

%% Set priors and settings
Aprior = 0.5; % flat priors
Rprior = 0.5;
Wprior = 0.5;

Wlambda = 50;

nSamples    = 100;
nRep        = 10;

Wlamda      = 4; % flat priors for inference
perceptlambda = 50; % precise priors for generation

%% Simulation parameters
gen_mu(:,1) = 0:0.1:2; gen_mu(:,2) = zeros(length(gen_mu),1);
gen_lambda  = 3;

gen_X       = nan(length(gen_mu),nRep,nSamples,2);
allSamples  = cell(length(gen_mu),nRep);

p_a         = nan(length(gen_mu),nRep);
p_r         = nan(length(gen_mu),nRep);
p_lambda    = nan(length(gen_mu),nRep);
p_w         = nan(length(gen_mu),3,nRep);
for m = 1:length(gen_mu)

        fprintf('Mu %d of %d \n',m,length(gen_mu))

    for i = 1:nRep        

        % generate data
        gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
        R = chol(gen_sigma);

        X = repmat(gen_mu(m,:),nSamples,1) + randn(nSamples,2)*R;
        gen_X(m,i,:,:) = X;

        % evaluate model
        samples = HOSS_evaluation_precision(X, Aprior, Rprior, Wprior, nSamples, Wlambda);
        allSamples{m,i} = samples;

        % Extract maximum a posteriori (MAP) estimates
        [f,xi] = ksdensity(samples.pA(:));
        [~,idx] = max(f);
        p_a(m,i) = xi(idx); % presence

        [f,xi] = ksdensity(samples.pR(:));
        [~,idx] = max(f);
        p_r(m,i) =  xi(idx); % reality

        [f,xi] = ksdensity(samples.senselambda(:));
        [~,idx] = max(f);
        p_lambda(m,i) = xi(idx); % continuous precision

        for w0 = 1:3
            tmp = samples.pW(:,:,w0);
            [f,xi] = ksdensity(tmp(:));
            [~,idx] = max(f);
            p_w(m,w0,i) = xi(idx);
        end

    end
end

% transform to p_w1 (relative to p_w2)
p_w1 = squeeze(p_w(:,2,:)./(p_w(:,2,:)+p_w(:,3,:)));

save(fullfile(root,'Model_perception','simulation_perception.mat'))
load(fullfile(root,'Model_perception','simulation_perception.mat'))

%% Plot the results
figure;

% awareness
subplot(1,2,1);
ac_map = makeColorMaps('maroon');
plotCI(p_a,gen_mu(:,1)','CI',ac_map(150,:),ac_map(150,:),'over');
hold on; plot(gen_mu(:,1),mean(p_a,2),'Color',ac_map(150,:),'LineWidth',2);

% perceptual inference
wc_map = makeColorMaps('teals'); hold on;
plotCI(p_w1,gen_mu(:,1)','CI',wc_map(150,:),wc_map(150,:),'over');
hold on; plot(gen_mu(:,1),mean(p_w1,2),'Color',wc_map(150,:),'LineWidth',2);

% reality judgement
subplot(1,2,2);
unc = 4; con = 17;
inferences = cat(3,p_w1([unc con],:),p_a([unc con],:), p_r([unc con],:));
barwitherr(squeeze(std(inferences,[],2))./sqrt(nRep),squeeze(mean(inferences,2)));
set(gca,'XTickLabel',{'Unconscious','Conscious'});
hold on; plot(xlim,[0.5 0.5])

