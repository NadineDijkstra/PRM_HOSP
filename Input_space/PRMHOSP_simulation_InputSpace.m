restoredefaultpath;
clc;
clear all;
addpath('D:\PRM_HOSP\Matlab_code_vector_var\Utilities')
addpath('D:\PRM_HOSP\Matlab_code_vector_var')
cd('D:\PRM_HOSP\Matlab_code_vector_var')

%% Set priors
Aprior = 0.5; % flat priors
Rprior = 0.5;
Wprior = 0.5;

Wlambda = 50;

%% Generate data and perform model inference
%  (X's from a given mean and precision)
% generate values of X from a bivariate normal distribution
gen_precision_vals = [1 3]; % 2 precision levels
gen_mu_vals        = 0:0.1:2;

nPrecision  = length(gen_precision_vals);
nMu         = length(gen_mu_vals);
nSamples    = 500;

p_a         = zeros(nMu,nMu,nPrecision);
p_r         = zeros(nMu,nMu,nPrecision);
p_lambda    = zeros(nMu,nMu,nPrecision);
p_w         = zeros(3,nMu,nMu,nPrecision);

% loop over the input values
allSamples = cell(nMu,nMu,nPrecision);
for mu1 = 1:nMu
    for mu2 = 1:nMu

        fprintf('mu1 %d, mu2 %d out of %d \n',mu1,mu2,nMu)

        for p = 1:nPrecision

            fprintf('\t precision val %d out of %d \n',p,nPrecision)

            gen_mu = [gen_mu_vals(mu1) gen_mu_vals(mu2)];
            gen_precision =  gen_precision_vals(p);
            gen_sigma = [1./gen_precision 0; 0 1./gen_precision];
            R = chol(gen_sigma);

            X = repmat(gen_mu,nSamples,1) + randn(nSamples,2)*R;

            % evaluate model
            samples = HOSS_evaluation_precision(X, Aprior, Rprior, Wprior, nSamples,Wlambda);
            allSamples{mu1,mu2,p} = samples;

            % extract posteriors
            p_a(mu1,mu2,p) = mean(samples.pA(:)); % presence
            p_r(mu1,mu2,p) = mean(samples.pR(:)); % reality
            p_lambda(mu1,mu2,p) = mean(samples.senselambda(:)); % continuous precision

            for w = 1:3
                tmp = samples.pW(:,:,w);
                p_w(w,mu1,mu2,p) = mean(tmp(:));
            end

        end

    end
end

cd('D:\PRM_HOSP\Matlab_code_vector_var\Input_Space');
save('simulation_input_precision_input_space.mat')
load('Input_Space/simulation_input_precision_input_space.mat')

%% Plot
c_map = makeColorMaps('teals');

% W-level poster
w_labels = {'w0','w1','w2'};
figure;
for w = 1:3
    for p = 1:nPrecision
        subplot(3,nPrecision+1,(w-1)*3+p)
        imagesc(gen_mu_vals,gen_mu_vals,squeeze(p_w(w,:,:,p)));
        caxis([0 1]); xlabel('gen mu2'); ylabel('gen mu1')
        title(sprintf('%s, precision %d',w_labels{w},p)); axis xy;
    end
end
colormap(c_map)

cs(1,:) = c_map(60,:);
cs(2,:) = c_map(120,:);
cs(3,:) = c_map(220,:);
for p = 1:nPrecision
    for w = 1:3
        if w==1
            % inverse diagonal
            tmp = squeeze(p_w(w,:,:,p));
            tmp2 = eye(nMu);
            pp = tmp(tmp2==1);
            xlab    = 'gen mu 1/2';
        elseif w==2
            % zero other mean
            pp = squeeze(p_w(w,:,1,p));
            if size(pp,1)==1; pp = pp'; end
            xlab    = 'gen mu 1';
        elseif w==3
            % zero other mean
            pp = squeeze(p_w(w,1,:,p));
            if size(pp,1)==1; pp = pp'; end
            xlab    = 'gen mu 2';
        end
        subplot(3,nPrecision+1,w*3)
        plotCI(pp,gen_mu_vals,'SEM',cs(p,:),cs(p,:),'over'); hold on;
        plot(gen_mu_vals',mean(pp,2),'Color',cs(p,:), 'LineWidth',2); hold on;
        ylim([0 1]); title(sprintf('w%d', w));
        xlabel(xlab)
    end
end

% A-level posterior - awareness
c_map = makeColorMaps('maroon');
figure;
for p = 1:nPrecision
    subplot(3,nPrecision+1,p)
    imagesc(gen_mu_vals,gen_mu_vals,squeeze(p_a(:,:,p)));
    axis xy; caxis([0 1]); xlabel('gen mu2'); ylabel('gen mu1') ;
    title(sprintf('A, precision %d',p))
end
colormap(c_map);

% along one axis
cs(1,:) = c_map(60,:);
cs(2,:) = c_map(120,:);
cs(3,:) = c_map(220,:);
for p = 1:nPrecision
    x = squeeze(p_a(:,:,p));
    pp = zeros(nMu,nReps);
    for r = 1:nReps
        tmp = squeeze(x(:,:,r));
        tmp2= eye(nMu);
        pp(:,r) = tmp(:,1); %tmp(tmp2==1);
    end
    subplot(3,nPrecision+1,nPrecision+1)
    hold on; plotCI(pp,gen_mu_vals,'SEM',cs(p,:),cs(p,:),'over'); hold on;
    plot(gen_mu_vals',mean(pp,2),'Color',cs(p,:), 'LineWidth',2);
    ylim([0 1]);
end

% R-level posterior - reality
for p = 1:nPrecision
    subplot(3,nPrecision+1,p+3)
    imagesc(gen_mu_vals,gen_mu_vals,squeeze(p_r(:,:,p)));
    axis xy; caxis([0 1]); xlabel('gen mu2'); ylabel('gen mu1');
    title(sprintf('R, precision %d',p))
end

% along one axis
colormap(c_map);
for p = 1:nPrecision
    x = squeeze(p_r(:,:,p));
    pp = zeros(nMu,nReps);
    for r = 1:nReps
        tmp = squeeze(x(:,:,r));
        tmp2= eye(nMu);
        pp(:,r) = tmp(:,1); %tmp(tmp2==1);
    end
    subplot(3,3,2*nPrecision+nPrecision)
    hold on; plotCI(pp,gen_mu_vals,'SEM',cs(p,:),cs(p,:),'over'); hold on;
    plot(gen_mu_vals',mean(pp,2),'Color',cs(p,:), 'LineWidth',2);
    ylim([0 1])
end

% lambda-level posterior - continuous precision
for p = 1:nPrecision
    subplot(3,nPrecision+1,p+(3*nPrecision))
    imagesc(gen_mu_vals,gen_mu_vals,squeeze(p_lambda(:,:,p)));
    axis xy; caxis([0 4]); xlabel('gen mu2'); ylabel('gen mu1')
end

colormap(c_map);
% along one axis
for p = 1:nPrecision
    x = squeeze(p_lambda(:,:,p));
    pp = zeros(nMu,nReps);
    for r = 1:nReps
        tmp = squeeze(x(:,:,r));
        tmp2= eye(nMu);
        pp(:,r) = tmp(:,1); %tmp(tmp2==1);
    end
    subplot(3,3,1+2*nPrecision+(nPrecision*2))
    plotCI(pp,gen_mu_vals,'SEM',cs(p,:),cs(p,:),'over'); hold on;
    plot(gen_mu_vals',mean(pp,2),'Color',cs(p,:), 'LineWidth',2); hold on;
    ylim([0 4])
end