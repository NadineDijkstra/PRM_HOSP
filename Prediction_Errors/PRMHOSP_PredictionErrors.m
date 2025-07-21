restoredefaultpath;
clc;
clear all;
root = 'D:\PRM_HOSS\Matlab_code_vector_var';
output_dir = 'Prediction_Errors';
addpath(fullfile(root))
addpath(fullfile(root,'Utilities'))
addpath(fullfile(root,output_dir))
cd(root)

%% Goal
% Compute prediction errors - defined as the KL divergence between the
% prior and posterior
% (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) - at
% different levels of the model for imagined and perceived perceptual
% content to see if PEs, and therefore model update, are lower for imagery

%% Settings
nSamples = 1667;
nChains  = 3;
nSamples = nChains*nSamples;
nReps    = 20;

%% Simulate KL divergence
% prior settings - perception, imagery, hallucination, Perky 
Aprior = [0.5 0.9 0.9 0.9]; App = [12 2000 2000 2000];
Rprior = [0.5 0.5 0.9 0.1]; Rpp = [12 12 2000 2000];
Wprior = [0.5 0.9 0.9 0.9]; perceptlambda = [4 50 50 50];

% define input
N = 100;
gen_lambda = [3 1 1 3]; % input precision
gen_mu     = [2 0; 0 0; 0 0; 2 0]; % input mu

nM = length(Aprior);

KL_divergence = nan(nReps,nM,6); % pA, pR, pW0, pW1, pW1, X
post_dist = cell(nM,6);
prior_dist = cell(nM,6);
for m = 1:nM % modality 

    fprintf('Simulation %d out of %d \n',m, nM)

    for r = 1:nReps

        fprintf('\t Repetition %d out of %d \n',r,nReps)

        % --- sample from the prior --- %
        [prior_samples] = HOSSPRM_sample_from_prior(Aprior(m),App(m),Rprior(m),Rpp(m),...
            Wprior(m),perceptlambda(m),nSamples);

        % --- sample from the posterior --- %
        gen_sigma = [1./gen_lambda(m) 0; 0 1./gen_lambda(m)];
        R = chol(gen_sigma);
        X = repmat(gen_mu(m,:),N,1) + randn(N,2)*R;

        % perform inference
        posterior_samples = HOSS_evaluation_precision(X, Aprior(m), Rprior(m), Wprior(m),...
            N, perceptlambda(m), App(m), Rpp(m));
        
        % --- calculate KL divergence --- %%
        top_levels = {'A','R'};
        for i = 1:length(top_levels)
            prior = eval(sprintf('prior_samples.p%s',top_levels{i}));            
            post  = eval(sprintf('posterior_samples.p%s',top_levels{i}));            
            
            [KL_divergence(r,m,i)] = calculate_KL_divergence(prior(:),post(:));

            % save distributions
            if r == 1
                post_dist{m,i} = post(:);
                prior_dist{m,i} = prior(:);
            else
                post_dist{m,i} = cat(1,post_dist{m,i},post(:));
                prior_dist{m,i} = cat(1,prior_dist{m,i},prior(:));
            end
        end

        low_levels = {'W'};
        for w = 1:3
            prior = eval(sprintf('prior_samples.p%s',low_levels{1}));
            prior = prior(:,w);
            post  = eval(sprintf('posterior_samples.p%s',low_levels{1}));
            post = post(:,:,w);
            KL_divergence(r,m,i+w) = calculate_KL_divergence(prior(:),post(:));

            % save distributions
            if r == 1
                post_dist{m,i+w} = post(:);
                prior_dist{m,i+w} = prior(:);
            else
                post_dist{m,i+w} = cat(1,post_dist{m,i+w},post(:));
                prior_dist{m,i+w} = cat(1,prior_dist{m,i+w},prior(:));
            end
        end

        % for X 
        prior = HOSSPRM_sample(Aprior(m), App(m),Rprior(m), Rpp(m), Wprior(m), perceptlambda(m), N);
        gen_sigma = [1./gen_lambda(m) 0; 0 1./gen_lambda(m)];
        R = chol(gen_sigma);
        posterior = repmat(gen_mu(m,:),N,1) + randn(N,2)*R;    
        KL_divergence(r,m,i+w+1) = calculate_KL_divergence(prior,posterior);      

        % save distributions
        if r == 1
            post_dist{m,i+w+1} = posterior;
            prior_dist{m,i+w+1} = prior;
        else
            post_dist{m,i+w+1} = cat(1,post_dist{m,i+w+1},posterior);
            prior_dist{m,i+w+1} = cat(1,prior_dist{m,i+w+1},prior);
        end

    end
end

% Plot distributions
parameters = {'pA','pR','pw0','pw1','pw2','X'};
experiences = {'Perception','Imagery','Hallucination','Perky'};
m_idx = [1 3 5 7 9; 2 4 6 8 10];
figure; 
for m = 1:nM
%     for p = 1:5 % parameters
%         subplot(6,nM,m+((p-1)*nM))
%         histogram(prior_dist{m,p}(:),'FaceColor','b','EdgeAlpha',0); hold on
%         histogram(post_dist{m,p}(:),'FaceColor','r','EdgeAlpha',0); hold on
%         title(sprintf('%s: %s',experiences{m},parameters{p}))
%         xlim([0 1])
%     end
    
    %p = 6;
    %subplot(6,nM*2,(5*nM*2)+m_idx(1,m))
    figure(2);
    subplot(2,nM,m)
    edges = linspace(-3, 3, 30+1);
    prior = histcounts2(prior_dist{m,p}(:,1), prior_dist{m,p}(:,2), edges, edges);
    imagesc(edges,edges, prior); axis xy; axis square; 
    title('prior');  caxis([0 1400]); 
    colorbar; 

    %subplot(6,nM*2,(5*nM*2)+m_idx(2,m))
    subplot(2,nM,m+nM)
    post = histcounts2(post_dist{m,p}(:,1), post_dist{m,p}(:,2), edges, edges);
    imagesc(edges,edges, post); axis xy; axis square; 
    title('posterior');  caxis([0 1400]); 
    colorbar; 
end

% Plot KL per level
figure; 
barwitherr(squeeze(std(KL_divergence))'./sqrt(nReps),squeeze(mean(KL_divergence))');
set(gca,'XTickLabels',parameters)
legend(experiences)



%% Simulate KL divergence at X 
nBins = 30;
edges = linspace(-3, 3, nBins+1);  % Bin edges from 0 to 1
plotting = true;
nReps = 100;
KL_divergence = nan(nReps,2,5); % pA, pR, pW0, pW1, pW1

% prior settings
Aprior = [0.5 0.9]; App = [12 200];
Rprior = [0.5 0.5]; Rpp = [12 12];
Wprior = [0.5 0.9]; perceptlambda = [4 50];

% define input
N = 500;
gen_lambda = [3 1]; 
gen_mu     = [2 0; 0 0]; 

v = VideoWriter('X_PEs.avi');  % Create video file
v.FrameRate = 2; open(v);
KL = nan(2,nReps); names = {'perception','imagery'};
priors = nan(nBins,nBins,2,nReps); posteriors = nan(nBins,nBins,2,nReps);
for r = 1:nReps
    figure(1);
    for m = 1:2

        % prior
        prior = HOSSPRM_sample(Aprior(m), App(m),Rprior(m), Rpp(m), Wprior(m), perceptlambda(m), N);        

        % posterior
        gen_sigma = [1./gen_lambda(m) 0; 0 1./gen_lambda(m)];
        R = chol(gen_sigma);
        posterior = repmat(gen_mu(m,:),N,1) + randn(N,2)*R;            

        % calculate divergence
        KL(m,r) = calculate_KL_divergence(prior,posterior);

        % plot priors/posteriors       
        [counts_prior, xedges, yedges] = histcounts2(prior(:,1), prior(:,2), edges, edges);
        priors(:,:,m,r) = counts_prior;

        [counts_post, xedges, yedges] = histcounts2(posterior(:,1), posterior(:,2), edges, edges);
        posteriors(:,:,m,r) = counts_post; 

        if plotting
            subplot(2,2,(m-1)*2+1);
            imagesc(xedges, yedges, counts_prior); axis xy; axis square; colorbar;
            title(sprintf('%s: prior',names{m})); xlim([-3 3]); ylim([-3 3]); caxis([0 8])
            subplot(2,2,m*2);
            imagesc(xedges, yedges, counts_post); axis xy; axis square; colorbar;
            title(sprintf('%s: posterior',names{m}));xlim([-3 3]); ylim([-3 3]); caxis([0 8])
        end
    end
    drawnow;

    % Capture the frame and write to video
    frame = getframe(gcf);
    writeVideo(v, frame);
end
close(v);

% Plot the posteriors/priors
figure(2);
KL_mean = nan(2,1);
for m = 1:2
    % mean over reps
    prior = squeeze(mean(priors(:,:,m,:),4));
    posterior = squeeze(mean(posteriors(:,:,m,:),4));

    % plot
    subplot(2,2,(m-1)*2+1); 
    imagesc(edges,edges, prior); axis xy; axis square; 
    colorbar; title('prior'); xlim([-3 3]); ylim([-3 3]); caxis([0 8])
    subplot(2,2,m*2);
    imagesc(edges, edges, posterior); axis xy; axis square; 
    colorbar; title('posterior');xlim([-3 3]); ylim([-3 3]); caxis([0 8])

    % to prevent zeroes
    epsilon = 1e-10;
    post = posterior + epsilon;
    prior = prior + epsilon;

    % renormalize
    post = post / sum(post(:));
    prior = prior / sum(prior(:));

    % calculate KL divergence
    KL_mean(m) = sum(post(:).*(log(post(:)./prior(:))));
end


