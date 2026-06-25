restoredefaultpath;
clc;
clear all;
root = [];
outDir = fullfile(root,'Model_comparison');
addpath(fullfile(fileparts(root),'Utilities'))
addpath(fullfile(outDir))
cd(outDir)


%% Perception and blindsight
% priors
Wprior = 0.5;
Wlambda = 4; 
Rprior = 0.5; 
Rpp    = 12;
Aprior = 0.5;
App    = 12;

% simulation settings
nSamples    = 100;
nRep        = 10;

% input parameters
gen_lambda  = 3;
gen_mu(:,1) = 0:0.1:2; gen_mu(:,2) = zeros(length(gen_mu),1);

% pre-allocate
p_w           = nan(length(gen_mu),3,nRep);
p_a           = nan(length(gen_mu),nRep);
p_r           = nan(length(gen_mu),nRep);

% run simulations
for m = 1:length(gen_mu)

    fprintf('mu %d of %d \n',m,length(gen_mu))

    for i = 1:nRep

        % generate data
        gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
        R = chol(gen_sigma);

        X = repmat(gen_mu(m,:),nSamples,1) + randn(nSamples,2)*R;
   
        % evaluate model
        samples = HOSS_evaluation_full(X, Aprior, Rprior, Wprior,nSamples, Wlambda, App, Rpp);

        % Extract maximum a posteriori (MAP) estimates
        tmp = samples.pA(:);
        [f,xi] = ksdensity(tmp(:));
        [~,idx] = max(f);
        p_a(m,i) = xi(idx);

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

save(fullfile(root,'Model_comparison','M3_perception.mat'))

% transform to p_w1 (relative to p_w2)
p_w1 = squeeze(p_w(:,2,:)./(p_w(:,2,:)+p_w(:,3,:)));

figure;

% awareness
a_color = [0 0 0.8];
plotCI(p_a,gen_mu(:,1)','CI',a_color,a_color,'over');
hold on; plot(gen_mu(:,1)',mean(p_a,2),'Color',a_color,'LineWidth',2);

% reality
ac_map = makeColorMaps('maroon');
r_color = ac_map(200,:);
plotCI(p_r,gen_mu(:,1)','CI',r_color,r_color,'over');
hold on; plot(gen_mu(:,1)',mean(p_r,2),'Color',r_color,'LineWidth',2);

% perceptual inference
wc_map = makeColorMaps('teals'); hold on;
plotCI(p_w1,gen_mu(:,1)','CI',wc_map(120,:),wc_map(120,:),'over');
hold on; plot(gen_mu(:,1),mean(p_w1,2),'Color',wc_map(120,:),'LineWidth',2);

hold on; plot(xlim,[0.5 0.5],'k--'); ylim([0 1])

%% Imagery without A prior modulation 
% priors
Wprior = 0.9;
Wlambdas = logspace(log10(4), log10(500), 21);
Rprior = 0.5; 
Rpp    = 12;
Aprior = 0.5; App = 12;
N = length(Wlambdas);

% simulation settings
nSamples    = 100;
nRep        = 10;

% input parameters
gen_lambda  = 1;
gen_mu      = [0 0];

% pre-allocate
p_w           = nan(N,3,nRep);
p_a           = nan(N,nRep);
p_r           = nan(N,nRep);

% run simulations
for m = 1:N

    fprintf('Aprior precision %d of %d \n',m,N)

    for i = 1:nRep

        % generate data
        gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
        R = chol(gen_sigma);

        X = repmat(gen_mu,nSamples,1) + randn(nSamples,2)*R;
   
        % evaluate model
        samples = HOSS_evaluation_full(X, Aprior, Rprior, Wprior, nSamples, Wlambdas(m), App, Rpp);

        % Extract maximum a posteriori (MAP) estimates
        tmp = samples.pA(:);
        [f,xi] = ksdensity(tmp(:));
        [~,idx] = max(f);
        p_a(m,i) = xi(idx);

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

save(fullfile(root,'Model_comparison','M3_ImageryOnlyW.mat'))

% transform to p_w1 (relative to p_w2)
p_w1 = squeeze(p_w(:,2,:)./(p_w(:,2,:)+p_w(:,3,:)));

figure;

% awareness
a_color = [0 0 0.8];
plotCI(p_a,Wlambdas,'CI',a_color,a_color,'over');
hold on; semilogx(Wlambdas,mean(p_a,2),'Color',a_color,'LineWidth',2);

% reality
ac_map = makeColorMaps('maroon');
r_color = ac_map(200,:);
plotCI(p_r,Wlambdas,'CI',r_color,r_color,'over');
hold on; semilogx(Wlambdas,mean(p_r,2),'Color',r_color,'LineWidth',2);

% perceptual inference
wc_map = makeColorMaps('teals'); hold on;
plotCI(p_w1,Wlambdas,'CI',wc_map(120,:),wc_map(120,:),'over');
hold on; semilogx(Wlambdas,mean(p_w1,2),'Color',wc_map(120,:),'LineWidth',2);

hold on; plot(xlim,[0.5 0.5],'k--'); ylim([0 1])
set(gca, 'XScale', 'log'); xlim([Wlambdas(1) Wlambdas(end)])

%% Imagery and aphantasia
% priors
Wprior = 0.9;
Wlambda = 500; 
Rprior = 0.5; 
Rpp    = 12;
Aprior = 0.9;
App    = logspace(log10(12),log10(5000),21);
N = length(App);

% simulation settings
nSamples    = 100;
nRep        = 10;

% input parameters
gen_lambda  = 1;
gen_mu      = [0 0];

% pre-allocate
p_w           = nan(N,3,nRep);
p_a           = nan(N,nRep);
p_r           = nan(N,nRep);

% run simulations
for m = 1:N

    fprintf('Aprior precision %d of %d \n',m,N)

    for i = 1:nRep

        % generate data
        gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
        R = chol(gen_sigma);

        X = repmat(gen_mu,nSamples,1) + randn(nSamples,2)*R;
   
        % evaluate model
        samples = HOSS_evaluation_full(X, Aprior, Rprior, Wprior, nSamples, Wlambda, App(m), Rpp);

        % Extract maximum a posteriori (MAP) estimates
        tmp = samples.pA(:);
        [f,xi] = ksdensity(tmp(:));
        [~,idx] = max(f);
        p_a(m,i) = xi(idx);

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

save(fullfile(root,'Model_comparison','M3_ImageryAlsoA.mat'))

% transform to p_w1 (relative to p_w2)
p_w1 = squeeze(p_w(:,2,:)./(p_w(:,2,:)+p_w(:,3,:)));

figure;

% awareness
a_color = [0 0 0.8];
plotCI(p_a,App,'CI',a_color,a_color,'over');
hold on; semilogx(App,mean(p_a,2),'Color',a_color,'LineWidth',2);

% reality
ac_map = makeColorMaps('maroon');
r_color = ac_map(200,:);
plotCI(p_r,App,'CI',r_color,r_color,'over');
hold on; semilogx(App,mean(p_r,2),'Color',r_color,'LineWidth',2);

% perceptual inference
wc_map = makeColorMaps('teals'); hold on;
plotCI(p_w1,App,'CI',wc_map(120,:),wc_map(120,:),'over');
hold on; semilogx(App,mean(p_w1,2),'Color',wc_map(120,:),'LineWidth',2);

hold on; plot(xlim,[0.5 0.5],'k--'); ylim([0 1])
set(gca, 'XScale', 'log'); xlim([App(1) App(end)])

%% PRM and individual differences in vividness 
% priors
Wprior = [0.9 0.9]; % same strength in low-level predictions
Wlambda = [500 500]; 
Rprior = [0.9 0.9]; % high prior on precision
Rpp    = [1000 1000];
Aprior = [0.9 0.9];
App    = [100 1000];
N = length(App);

% simulation settings
nSamples    = 100;
nRep        = 10;

% input parameters
gen_lambda  = 1;
gen_mu      = [0 0];

% pre-allocate
allSamples    = cell(N,nRep);
p_w           = nan(N,3,nRep);
p_a           = nan(N,nRep);
p_r           = nan(N,nRep);


% run simulations
for m = 1:N

    fprintf('Aprior precision %d of %d \n',m,N)

    for i = 1:nRep

        % generate data
        gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
        R = chol(gen_sigma);

        X = repmat(gen_mu,nSamples,1) + randn(nSamples,2)*R;
   
        % evaluate model
        samples = HOSS_evaluation_full(X, Aprior(m), Rprior(m), Wprior(m), nSamples, Wlambda(m), App(m), Rpp(m));

        % Extract maximum a posteriori (MAP) estimates
        tmp = samples.pA(:);
        [f,xi] = ksdensity(tmp(:));
        [~,idx] = max(f);
        p_a(m,i) = xi(idx);

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

save(fullfile(root,'Model_comparison','M3_IndividualDifferences.mat'))

% transform to p_w1 (relative to p_w2)
p_w1 = squeeze(p_w(:,2,:)./(p_w(:,2,:)+p_w(:,3,:)));

% plot
figure; 
for m = 1:2
    subplot(1,2,m);
    dat = [squeeze(p_w1(m,:))' squeeze(p_r(m,:))' squeeze(p_a(m,:))'];
    M   = mean(dat,1); SEM = std(dat)./sqrt(N);
    barwitherr(SEM,M); set(gca,'XTickLabel',{'w','R','A'}); ylim([0 1])
end

%% Dendritic Information Theory congruency hypothesis
% Priors
Wprior = [0.9 0.1]; % congruent vs incongruent W level
Wlambda = 500; 

Rprior = 0.5; % A and R flat
Rpp    = 12;
Aprior = 0.5;
App    = 12;

N = length(Wprior);

% simulation settings
nSamples    = 100;
nRep        = 10;

% input parameters
gen_lambda  = 3;
gen_mu      = [0.5 0; 1 0; 1.5 0; 2 0];
nMu         = size(gen_mu,1);

% pre-allocate
p_w           = nan(N,nMu,3,nRep);
p_a           = nan(N,nMu,nRep);
p_r           = nan(N,nMu,nRep);

% run simulations
for w = 1:N

    for m = 1:nMu

        fprintf('Prior %d - input %d \n',w,m)

        for i = 1:nRep

            % generate data
            gen_sigma = [1./gen_lambda 0; 0 1./gen_lambda];
            R = chol(gen_sigma);

            X = repmat(gen_mu(m,:),nSamples,1) + randn(nSamples,2)*R;

            % evaluate model
            samples = HOSPPRM_evaluation_full(X, Aprior, Rprior, Wprior(w), nSamples, Wlambda, App, Rpp);

            % Extract maximum a posteriori (MAP) estimates
            tmp = samples.pA(:);
            [f,xi] = ksdensity(tmp(:));
            [~,idx] = max(f);
            p_a(w,m,i) = xi(idx);

            tmp = samples.pR(:);
            [f,xi] = ksdensity(tmp(:));
            [~,idx] = max(f);
            p_r(w,m,i) = xi(idx);

            for w0 = 1:3
                tmp = samples.pW(:,:,w0);
                [f,xi] = ksdensity(tmp(:));
                [~,idx] = max(f);
                p_w(w,m,w0,i) = xi(idx);
            end

        end
    end
end

% Plot the results
figure;
SEM = squeeze(std(p_a,[],3))./sqrt(nRep);
M   = squeeze(mean(p_a,3));
barwitherr(SEM',M'); 
set(gca,'XTickLabels',{'[0.5 0]','[1 0]','[1.5 0]','[2 0]'})
legend('Congruent','Incongruent')

