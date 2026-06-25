restoredefaultpath;
clc;
clear all;
root = [];
outDir = fullfile(root,'Model_comparison');
addpath(fullfile(fileparts(root),'Utilities'))
addpath(fullfile(outDir))
cd(outDir)

%% Perceptual inference 
% priors
Wprior = [1/3 1/3 1/3];

Wlambda       = 4; % flat priors for inference

% simulation settings
nSamples    = 100;
nRep        = 10;

% input parameters
gen_lambda  = 3;
gen_mu(:,1) = 0:0.1:2; gen_mu(:,2) = zeros(length(gen_mu),1);
senselambda = gen_lambda; % fixed precision 

% pre-allocate
allSamples    = cell(length(gen_mu),nRep);
p_w           = nan(length(gen_mu),3,nRep);
KL_divergence = nan(length(gen_mu),3,nRep); % per w state

% run simulations
for m = 1:length(gen_mu)

    fprintf('Mu %d of %d \n',m,length(gen_mu))

    for i = 1:nRep

        % Observe data
        gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
        R = chol(gen_sigma);
        X = repmat(gen_mu(m,:),nSamples,1) + randn(nSamples,2)*R;

        % Generate data from prior
        [~,priorW] = HOSSPRM_sample_firstorder_threestates(Wprior, Wlambda, gen_lambda,1667*3);        

        % Evaluate model
        samples = HOSS_evaluation_firstorder(X, Wprior, Wlambda, senselambda, nSamples);
        allSamples{m,i} = samples;

        % Extract maximum a posteriori (MAP) estimates
        for w0 = 1:3
            tmp = samples.pW(:,:,w0);
            [f,xi] = ksdensity(tmp(:));
            [~,idx] = max(f);
            p_w(m,w0,i) = xi(idx);
        end

        % Calculate prediction errors
        for w0 = 1:3
            postW = squeeze(samples.pW(:,:,w0));
            KL_divergence(m,w0,i) = calculate_KL_divergence(priorW(:,w0),postW(:));

            % Save distributions
            if i == 1
                post_dist{m,w0} = postW;

                prior_dist{m,w0} = priorW(:,w0);
            else
                post_dist{m,w0} = cat(1,post_dist{m,w0},postW);

                prior_dist{m,w0} = cat(1,prior_dist{m,w0},priorW(:,w0));
            end
        end

    end
end

save(fullfile(root,'Model_comparison','M1_perception.mat'))


%% Mental imagery (prior driven)
% priors
Wprior = [0.1/2 0.9 0.1/2];

Wlambdas       = logspace(log10(4), log10(500), 21); % strong priors for imagery 

% simulation settings
nSamples    = 100;
nRep        = 10;

% input parameters
gen_lambda  = 1;
gen_mu      = [0 0];
senselambda = gen_lambda; % fixed precision 


% pre-allocate
gen_X       = nan(length(gen_mu),nRep,nSamples,2);
allSamples  = cell(length(gen_mu),nRep);

p_w         = nan(length(gen_mu),3,nRep);
KL_divergence = nan(length(gen_mu),nRep); % only for w1 (w2 is equivalent)

% run simulations
for w = 1:length(Wlambdas)

    fprintf('w %d of %d \n',w,length(Wlambdas))

    for i = 1:nRep

        % generate data
        gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
        R = chol(gen_sigma);

        X = repmat(gen_mu,nSamples,1) + randn(nSamples,2)*R;
        gen_X(w,i,:,:) = X;

        % Generate data from prior
        [~,priorW] = HOSSPRM_sample_firstorder_threestates(Wprior, Wlambdas(w), gen_lambda,1667*3);

        % evaluate model
        samples = HOSS_evaluation_firstorder(X, Wprior, Wlambdas(w), senselambda, nSamples);
        allSamples{w,i} = samples;

        % Extract maximum a posteriori (MAP) estimates
        for w0 = 1:2
            tmp = samples.pW(:,:,w0);
            [f,xi] = ksdensity(tmp(:));
            [~,idx] = max(f);
            p_w(w,w0,i) = xi(idx);
        end

        % Calculate prediction errors
        for w0 = 1:3
            postW = squeeze(samples.pW(:,:,w0));
            KL_divergence(w,w0,i) = calculate_KL_divergence(priorW(:,w0),postW(:));

            % Save distributions
            if i == 1
                post_dist{w,w0} = postW;

                prior_dist{w,w0} = priorW(:,w0);
            else
                post_dist{w,w0} = cat(1,post_dist{w,w0},postW);

                prior_dist{w,w0} = cat(1,prior_dist{w,w0},priorW(:,w0));
            end
        end

    end
end

save(fullfile(root,'Model_comparison','M1_imagery.mat'))

%% Plot inferences
wc_map = makeColorMaps('teals'); 
wc_idx = [40 120 220];

figure; 
% perception 
load(fullfile(root,'Model_comparison','M1_perception.mat'))
subplot(1,2,1);
for w = 1:3
    plotCI(squeeze(p_w(:,w,:)),gen_mu(:,1)','CI',...
        wc_map(wc_idx(w),:),wc_map(wc_idx(w),:),'over'); hold on;
    plot(gen_mu(:,1),squeeze(mean(p_w(:,w,:),3)),'Color',...
        wc_map(wc_idx(w),:),'LineWidth',2)
end
xlabel('x1'); title('Sensory input strength'); 

% imagery 
load(fullfile(root,'Model_comparison','M1_imagery.mat'))
subplot(1,2,2);
for w = 1:3
    plotCI(squeeze(p_w(:,w,:)),Wlambdas,'CI',...
        wc_map(wc_idx(w),:),wc_map(wc_idx(w),:),'over'); hold on;
    %plot(Wlambdas,squeeze(mean(p_w(:,w,:),3)),'Color',...
    %    wc_map(wc_idx(w),:),'LineWidth',2); hold on;
    semilogx(Wlambdas,squeeze(mean(p_w(:,w,:),3)),'Color',...
        wc_map(wc_idx(w),:),'LineWidth',2); hold on;
end
set(gca, 'XScale', 'log');  
ylim([0 1]); xlim([min(Wlambdas) max(Wlambdas)]);
xlabel('Prior precision'); title('Imagery')

%% Plot prediction errors 
% perception 
load(fullfile(root,'Model_comparison','M1_perception.mat'))

figure(1);
% prediction errors
for w0 = 1:3
    subplot(3,1,w0);
    plotCI(squeeze(KL_divergence(:,w0,:)),gen_mu(:,1)','CI',...
        wc_map(80,:),wc_map(80,:),'over'); hold on;
    plot(gen_mu(:,1),squeeze(mean(KL_divergence(:,w0,:),3)),'Color',...
        wc_map(80,:),'LineWidth',2);
    xlabel('gen mu x1'); ylabel('Prediction error'); %ylim([0 20])
    title(sprintf('w state %d',w0-1))
end

figure(2);
% priors and posterior
m_idx = round(linspace(1,21,4));
for m = 1:4
    for w0 = 1:3
        subplot(3,4,m+(w0-1)*4); edges = linspace(0,1,100);
        histogram(prior_dist{m_idx(m),w0}(:),edges,'FaceColor','b','EdgeAlpha',0); hold on
        histogram(post_dist{m_idx(m),w0}(:),edges,'FaceColor','r','EdgeAlpha',0); hold on
        xlim([0 1]); ylim([0 18000])
    end
end


% imagery
load(fullfile(root,'Model_comparison','M1_imagery.mat'))

figure(3);
% prediction errors
for w0 = 1:3
    subplot(3,1,w0);
    plotCI(squeeze(KL_divergence(:,w0,:)),Wlambdas,'CI',...
        wc_map(80,:),wc_map(80,:),'over'); hold on;
    semilogx(Wlambdas,squeeze(mean(KL_divergence(:,w0,:),3)),'Color',...
        wc_map(80,:),'LineWidth',2);
    set(gca, 'XScale', 'log');  
    xlim([min(Wlambdas) max(Wlambdas)]);
    xlabel('Prior precision'); ylabel('Prediction error'); %ylim([10 25])
    title(sprintf('w state %d',w0-1))    
end

figure(4);
% priors and posterior
m_idx = round(linspace(1,21,4));
for m = 1:4
    for w0 = 1:3
        subplot(3,4,m+(w0-1)*4); edges = linspace(0,1,100);
        histogram(prior_dist{m_idx(m),w0}(:),edges,'FaceColor','b','EdgeAlpha',0); hold on
        histogram(post_dist{m_idx(m),w0}(:),edges,'FaceColor','r','EdgeAlpha',0); hold on
        xlim([0 1]); ylim([0 18000])
    end
end





