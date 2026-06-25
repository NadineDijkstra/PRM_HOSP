% demonstrate the regime at which the effects of interest are obtained and
% at which the model fails
restoredefaultpath;
clc;
clear;
root = 'D:\PRM_HOSS\Matlab_code_vector_var\Revision_NatComm2025';
outDir = fullfile(root,'RobustnessAnalyses'); if ~exist(outDir,'dir'); mkdir(outDir); end
cd(root)
addpath(fullfile(fileparts(root),'Utilities'))
addpath(fullfile(root,'Model_comparison'))
cd(fullfile(root,'Model_comparison'))

%% Model 1: W-level prior for top-down imagery 
% priors
Wpriors  = linspace(0.5,1,6);
Wlambdas = logspace(log10(4), log10(500), 10); % strong priors for imagery 
W1       = length(Wpriors);
W2       = length(Wlambdas);

% simulation settings
nSamples    = 100;
nRep        = 10;

% input parameters
gen_lambda  = 1;
gen_mu      = [0 0];

% pre-allocate
p_w         = nan(W1,W2,3,nRep);
p_r         = nan(W1,W2,nRep);

% run simulations
for w1 = 1:W1 % prior
    for w2 = 1:W2 % prior predicions

        fprintf('w1 %d - w2 %d \n',w1,w2)

        for i = 1:nRep

            % generate data
            gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
            R = chol(gen_sigma);

            X = repmat(gen_mu,nSamples,1) + randn(nSamples,2)*R;

            % evaluate model
            Wprior = nan(1,3); Wprior(2) = Wpriors(w1); Wprior([1,3]) = (1-Wpriors(w1))/2;
            samples = HOSS_evaluation_firstorder(X, Wprior, Wlambdas(w2), gen_lambda, nSamples);

            % Extract maximum a posteriori (MAP) estimates
            for w0 = 1:3
                tmp = samples.pW(:,:,w0);
                [f,xi] = ksdensity(tmp(:));
                [~,idx] = max(f);
                p_w(w1,w2,w0,i) = xi(idx);
            end

        end
    end
end

save(fullfile(outDir,'M1_imagery.mat'))

figure;
for w = 1:3
    results = squeeze(mean(p_w(:,:,w,:),4)); % mean over reps
    subplot(1,3,w); imagesc(results'); caxis([0 1])  
    title(sprintf('w%d',w-1));
    xlabel('Wprior'); ylabel('W lambda')
end
colormap(makeColorMaps('teals'))

%% Model 2: R level prior for overestimating precision during imagery 
% priors
Wprior = [0.1/2 0.9 0.1/2];
Wlambdas = [5 50 500]; % three levels of W precision

Rprior = linspace(0.5,1,6); Rprior(end) = 0.95; % several levels of R strength and precision
Rpp    = logspace(log10(12), log10(1000), 10); 

nW     = length(Wlambdas); Rs = length(Rprior); Rp = length(Rpp);

% simulation settings
nSamples    = 100;
nRep        = 10;

% input parameters
gen_lambda  = 1;
gen_mu      = [0 0];

% pre-allocate
p_w         = nan(nW,Rs,Rp,3,nRep);
p_r         = nan(nW,Rs,Rp,nRep);

% run simulations
for w = 1:nW

    for r1 = 1:Rs

        for r2 = 1:Rp

            fprintf('w %d - rs %d - rp %d \n',w,r1,r2)

            for i = 1:nRep

                % generate data
                gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
                R = chol(gen_sigma);

                X = repmat(gen_mu,nSamples,1) + randn(nSamples,2)*R;

                % evaluate model
                try
                    samples = HOSS_evaluation_onlyprecision(X, Rprior(r1), Wprior, nSamples, Wlambdas(w), Rpp(r2));

                    % Extract maximum a posteriori (MAP) estimates
                    tmp = samples.pR(:);
                    [f,xi] = ksdensity(tmp(:));
                    [~,idx] = max(f);
                    p_r(w,r1,r2,i) = xi(idx);

                    for w0 = 1:3
                        tmp = samples.pW(:,:,w0);
                        [f,xi] = ksdensity(tmp(:));
                        [~,idx] = max(f);
                        p_w(w,r1,r2,w0,i) = xi(idx);
                    end
                catch

                end
            end
        end
    end
end

save(fullfile(outDir,'M2_topdownhallucination'))

% plot the results
figure;
for w = 1:3
    subplot(1,3,w);
    imagesc(squeeze(mean(p_r(w,:,:,:),4))'); caxis([0 1])  
    title(sprintf('w%d',w));
    xlabel('Rprior'); ylabel('R lambda')
end
colormap(makeColorMaps('maroon'))

%% Model 2: R level prior for understimating precision during perception 
% priors
Wprior = [0.1/2 0.9 0.1/2];
Wlambdas = [5 50 500]; % three levels of W precision

Rprior = linspace(0,0.5,6); % several levels of R strength and precision
Rpp    = logspace(log10(12), log10(1000), 10); 

nW     = length(Wlambdas); Rs = length(Rprior); Rp = length(Rpp);

% simulation settings
nSamples    = 100;
nRep        = 10;

% input parameters
gen_lambda  = 3;
gen_mu      = [1 0];

% pre-allocate
p_w         = nan(nW,Rs,Rp,3,nRep);
p_r         = nan(nW,Rs,Rp,nRep);

% run simulations
for w = 1:nW

    for r1 = 1:Rs

        for r2 = 1:Rp

            fprintf('w %d - rs %d - rp %d \n',w,r1,r2)

            for i = 1:nRep

                % generate data
                gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
                R = chol(gen_sigma);

                X = repmat(gen_mu,nSamples,1) + randn(nSamples,2)*R;

                % evaluate model
                try
                    samples = HOSS_evaluation_onlyprecision(X, Rprior(r1), Wprior, nSamples, Wlambdas(w), Rpp(r2));

                    % Extract maximum a posteriori (MAP) estimates
                    tmp = samples.pR(:);
                    [f,xi] = ksdensity(tmp(:));
                    [~,idx] = max(f);
                    p_r(w,r1,r2,i) = xi(idx);

                    for w0 = 1:3
                        tmp = samples.pW(:,:,w0);
                        [f,xi] = ksdensity(tmp(:));
                        [~,idx] = max(f);
                        p_w(w,r1,r2,w0,i) = xi(idx);
                    end
                catch

                end
            end
        end
    end
end

save(fullfile(outDir,'M2_Perkyeffect'))
load(fullfile(outDir,'M2_Perkyeffect'))

% plot the results
figure;
for w = 1:3
    subplot(1,3,w);
    imagesc(squeeze(mean(p_r(w,:,:,:),4))'); caxis([0 1])  
    title(sprintf('w%d',w));
    xlabel('Rprior'); ylabel('R lambda')
end
colormap(makeColorMaps('maroon'))

%% Model 3: A level prior for modelling conscious imagery
% priors
Wprior = 0.9;
Wlambdas = [5 50 500]; % three levels of W precision

Rprior = 0.5; % several levels of R strength and precision
Rpp    = 12; 

Aprior = linspace(0.5,1,6);
App    = logspace(log10(12), log10(5000), 10); 

nW     = length(Wlambdas); As = length(Aprior); Ap = length(App);

% simulation settings
nSamples    = 100;
nRep        = 10;

% input parameters
gen_lambda  = 1;
gen_mu      = [0 0];

% pre-allocate
p_w         = nan(nW,As,Ap,3,nRep);
p_r         = nan(nW,As,Ap,nRep);
p_a         = nan(nW,As,Ap,nRep);

% run simulations
for w = 1:nW

    for a1 = 1:As

        for a2 = 1:Ap

            fprintf('w %d - as %d - ap %d \n',w,a1,a2)

            for i = 1:nRep

                % generate data
                gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
                R = chol(gen_sigma);

                X = repmat(gen_mu,nSamples,1) + randn(nSamples,2)*R;

                % evaluate model
                try
                    samples = HOSS_evaluation_precision(X, Aprior(a1), Rprior, Wprior, nSamples, Wlambdas(w), App(a2), Rpp);

                    % Extract maximum a posteriori (MAP) estimates
                    tmp = samples.pR(:);
                    [f,xi] = ksdensity(tmp(:));
                    [~,idx] = max(f);
                    p_r(w,a1,a2,i) = xi(idx);

                    tmp = samples.pA(:);
                    [f,xi] = ksdensity(tmp(:));
                    [~,idx] = max(f);
                    p_a(w,a1,a2,i) = xi(idx);

                    for w0 = 1:3
                        tmp = samples.pW(:,:,w0);
                        [f,xi] = ksdensity(tmp(:));
                        [~,idx] = max(f);
                        p_w(w,a1,a2,w0,i) = xi(idx);
                    end
                catch

                end
            end
        end
    end
end

save(fullfile(outDir,'M3_ConsciousImagery'))
load(fullfile(outDir,'M3_ConsciousImagery'))

% plot the results
figure;
for w = 1:3
    subplot(1,3,w);
    imagesc(squeeze(mean(p_a(w,:,:,:),4))'); caxis([0 1])  
    title(sprintf('w%d',w));
    xlabel('A prior'); ylabel('A lambda')
end
map = ones(255,3); map(:,1:2) = flipud([linspace(0,1,255)' linspace(0,1,255)']);
colormap(map)