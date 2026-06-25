restoredefaultpath;
clc;
clear all;
root = []
outDir = fullfile(root,'Model_comparison');
addpath(fullfile(fileparts(root),'Utilities'))
addpath(fullfile(outDir))
cd(outDir)

%% Perceptual inference 
% priors
Wprior = [1/3 1/3 1/3];
Wlambda = 4; % flat priors for inference
Rprior = 0.5; 
Rpp    = 12;

% simulation settings
nSamples    = 100;
nRep        = 10;

% input parameters
gen_lambda  = 3;
gen_mu(:,1) = 0:0.1:2; gen_mu(:,2) = zeros(length(gen_mu),1);

% pre-allocate
allSamples    = cell(length(gen_mu),nRep);
p_w           = nan(length(gen_mu),3,nRep);
p_r           = nan(length(gen_mu),nRep);

% run simulations
for m = 1:length(gen_mu)

    fprintf('Mu %d of %d \n',m,length(gen_mu))

    for i = 1:nRep

        % Observe data
        gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
        R = chol(gen_sigma);
        X = repmat(gen_mu(m,:),nSamples,1) + randn(nSamples,2)*R;

        % Evaluate model
        samples = HOSS_evaluation_onlyprecision(X, Rprior, Wprior,nSamples, Wlambda, Rpp);
        allSamples{m,i} = samples;

        % Extract maximum a posteriori (MAP) estimates
        tmp = samples.pR(:);
        [f,xi] = ksdensity(tmp(:));
        [~,idx] = max(f);
        p_r(m,i) = xi(idx);

        for w0 = 1:3
            tmp = samples.pW(:,:,w0);
            [f,xi] = ksdensity(tmp(:));
            [~,idx] = max(f);
            p_w(m,w0,i) = xi(idx);
        end      

    end
end

save(fullfile(root,'Model_comparison','M2_perception.mat'))


%% Mental imagery (prior driven)
% priors
Wprior = [0.1/2 0.9 0.1/2];
Wlambdas = logspace(log10(4), log10(500), 21); % strong priors for imagery 
Rprior = 0.5; 
Rpp    = 12;

% simulation settings
nSamples    = 100;
nRep        = 10;

% input parameters
gen_lambda  = 1;
gen_mu      = [0 0];


% pre-allocate
allSamples  = cell(length(gen_mu),nRep);
p_w         = nan(length(gen_mu),3,nRep);
p_r           = nan(length(gen_mu),nRep);

% run simulations
for w = 1:length(Wlambdas)

    fprintf('w %d of %d \n',w,length(Wlambdas))

    for i = 1:nRep

        % generate data
        gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
        R = chol(gen_sigma);

        X = repmat(gen_mu,nSamples,1) + randn(nSamples,2)*R;
   
        % evaluate model
        samples = HOSS_evaluation_onlyprecision(X, Rprior, Wprior,nSamples, Wlambdas(w), Rpp);
        allSamples{w,i} = samples;

        % Extract maximum a posteriori (MAP) estimates
        tmp = samples.pR(:);
        [f,xi] = ksdensity(tmp(:));
        [~,idx] = max(f);
        p_r(w,i) = xi(idx);

        for w0 = 1:3
            tmp = samples.pW(:,:,w0);
            [f,xi] = ksdensity(tmp(:));
            [~,idx] = max(f);
            p_w(w,w0,i) = xi(idx);
        end  

    end
end

save(fullfile(root,'Model_comparison','M2_imagery.mat'))


%% Plot inferences

% perception 
load(fullfile(root,'Model_comparison','M1_perception.mat'))

figure; 
subplot(1,2,1);
wc_map = makeColorMaps('teals'); 
wc_idx = [40 120 220];
for w = 1:3
    plotCI(squeeze(p_w(:,w,:)),gen_mu(:,1)','CI',...
        wc_map(wc_idx(w),:),wc_map(wc_idx(w),:),'over'); hold on;
    plot(gen_mu(:,1),squeeze(mean(p_w(:,w,:),3)),'Color',...
        wc_map(wc_idx(w),:),'LineWidth',2)
end
rc_map = makeColorMaps('maroon'); 
for w = 1:3
    plotCI(p_r,gen_mu(:,1)','CI',...
        rc_map(200,:),rc_map(200,:),'over'); hold on;
    plot(gen_mu(:,1),squeeze(mean(p_r,2)),'Color',...
        rc_map(200,:),'LineWidth',2)
end
xlabel('x1'); title('Sensory input strength'); ylim([0 1])

% imagery
load(fullfile(root,'Model_comparison','M2_imagery.mat'))

subplot(1,2,2);
wc_map = makeColorMaps('teals'); 
wc_idx = [40 120 220];
for w = 1:3
    plotCI(squeeze(p_w(:,w,:)),Wlambdas,'CI',...
        wc_map(wc_idx(w),:),wc_map(wc_idx(w),:),'over'); hold on;
    semilogx(Wlambdas,squeeze(mean(p_w(:,w,:),3)),'Color',...
        wc_map(wc_idx(w),:),'LineWidth',2);
    set(gca, 'XScale', 'log');  
    xlim([min(Wlambdas) max(Wlambdas)]);
end
rc_map = makeColorMaps('maroon'); 
for w = 1:3
    plotCI(p_r,Wlambdas,'CI',...
        rc_map(200,:),rc_map(200,:),'over'); hold on;
    plot(Wlambdas,squeeze(mean(p_r,2)),'Color',...
        rc_map(200,:),'LineWidth',2)
end
xlabel('lambda'); title('Prior precision'); ylim([0 1])

