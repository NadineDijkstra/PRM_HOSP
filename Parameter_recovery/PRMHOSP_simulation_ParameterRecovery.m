restoredefaultpath;
clc;
clear all;
root = 'D:\PRM_HOSP\Matlab_code_vector_var';
addpath(fullfile(root))
addpath(fullfile(root,'Utilities'))
addpath(fullfile(root,'Parameter_Recovery'))
cd(root)

%% Parameters to recover
Wpriors = 0:0.1:1;
Apriors = 0:0.1:1;
Rpriors = 0:0.1:1;

%% Perform parameter recovery
nSamples    = 100;
nRep        = 10; 

Wlamda      = 50; 
perceptlambda = 50; 
nK          = length(Wpriors);

p_w         = zeros(3,nK,nK,nK,nRep);
p_a         = zeros(nK,nK,nK,nRep);
p_r         = zeros(nK,nK,nK,nRep);
p_lambda    = zeros(nK,nK,nK,nRep);

allSamples  = cell(nK,nK,nK,nRep);

for w = 1:nK    
    fprintf('Parameter recovery for w: %d \n',w)
    for a = 1:nK
        for r = 1:nK
            fprintf('\t a: %d, r: %d \n',a,r)

            for i = 1:nRep

                % Generate X
                X = HOSS_PRM_sample(Apriors(a), Rpriors(r), perceptlambda, Wpriors(w), nSamples);

                % Infer parameters
                samples = HOSS_evaluation_precision(X, 0.5, 0.5, 0.5, nSamples,Wlamda); % infer under flat priors
                %allSamples{w,a,r} = samples;

                % Extract maximum a posteriori (MAP) estimates
                [f,xi] = ksdensity(samples.pA(:));
                [~,idx] = max(f);
                p_a(w,a,r,i) = xi(idx); % presence

                [f,xi] = ksdensity(samples.pR(:));
                [~,idx] = max(f);
                p_r(w,a,r,i) =  xi(idx); % reality
                
                [f,xi] = ksdensity(samples.senselambda(:));
                [~,idx] = max(f);
                p_lambda(w,a,r,i) = xi(idx); % continuous precision

                for w0 = 1:3                    
                    tmp = samples.pW(:,:,w0);
                    [f,xi] = ksdensity(tmp(:));
                    [~,idx] = max(f);
                    p_w(w0,w,a,r,i) = xi(idx);
                end
            end
        end
    end
end

%% Save results
save(fullfile(root,'Parameter_Recovery','simulation_parameter_recovery'))
load(fullfile(root,'Parameter_Recovery','simulation_parameter_recovery'))
            
%% Plot the results
% full plots first 
p_w = squeeze(mean(p_w,5));

% W-level inferences - all parameters
c_map = makeColorMaps('teals');
for w0 = 1:3
    figure(w0);
    c = 1;
    for r = 1:nK
        tmp = squeeze(p_w(w0,:,:,r));
        subplot(4,3,c)
        imagesc(Wpriors,Apriors,tmp);
        caxis([0 1]); 
        axis xy; 
        xlabel('Aprior'); ylabel('Wprior');
        title(sprintf('W: %.1f, Precision: %.1f',w0-1,Rpriors(r)));
        c = c+1;
    end
    colormap(c_map)
end

% W-level inferences - influence of A/R

figure; % influence of A
A_idx = 1:11;
cs    = c_map(round(linspace(60,256,length(A_idx))),:);
for w0 = 1:3

    subplot(1,3,w0)
    for a = 1:length(A_idx)
        tmp = squeeze(p_w(w0,:,A_idx(a),:)); % mean over precision

        plotCI(tmp,Wpriors,'SEM',cs(a,:),cs(a,:),'over'); hold on;
        plot(Wpriors,mean(tmp,2),'Color',cs(a,:), 'LineWidth',2); hold on;        
    end
    ylim([-0.1 1.1]); title(sprintf('w%d', w0-1));
    xlabel('gen W priors')
end

figure; % influence of R
R_idx = 1:11;
cs    = c_map(round(linspace(60,256,length(R_idx))),:);
for w0 = 1:3

    subplot(1,3,w0)
    for r = 1:length(R_idx)
        tmp = squeeze(p_w(w0,:,:,R_idx(r))); % mean over precision

        plotCI(tmp,Wpriors,'SEM',cs(r,:),cs(r,:),'over'); hold on;
        plot(Wpriors,mean(tmp,2),'Color',cs(r,:), 'LineWidth',2); hold on;        
    end
    ylim([-0.1 1.1]); title(sprintf('w%d', w0-1));
    xlabel('gen W priors')
end

% A/R-level inference - all parameters
p_a = squeeze(mean(p_a,4));
p_r = squeeze(mean(p_r,4));
c_map = makeColorMaps('maroon');

figure; % A inference
c = 1;
for r = 1:nK
    tmp = squeeze(p_a(:,:,r));
    subplot(4,3,c)
    imagesc(Wpriors,Apriors,tmp);
    caxis([0 1]); axis xy;
    xlabel('Aprior'); ylabel('Wprior');
    title(sprintf('A, Precision: %.1f',Rpriors(r)));
    c = c+1;
end
colormap(c_map)

figure; % R inference
c = 1;
for a = 1:nK
    tmp = squeeze(p_r(:,a,:)); % rows are w, columnds are r
    subplot(4,3,c)
    imagesc(Wpriors,Rpriors,tmp);
    caxis([0 1]); axis xy
    xlabel('Rprior'); ylabel('Wprior');
    title(sprintf('R, A: %.1f',Apriors(a)));
    c = c+1;
end
colormap(c_map)

% influence of other params on A/R
figure; 
subplot(1,2,1);
W_idx = 1:11;
cs    = c_map(round(linspace(60,230,length(W_idx))),:);
for w = 1:length(W_idx)
    tmp = squeeze(p_a(W_idx(w),:,:)); 

    plotCI(tmp,Apriors,'SEM',cs(w,:),cs(w,:),'over'); hold on;
    plot(Apriors,mean(tmp,2),'Color',cs(w,:), 'LineWidth',2); hold on;
end
ylim([-0.1 1.1]); title(sprintf('w%d', w0-1));
xlabel('gen A priors'); ylabel('Inference A'); 
title('Influence of W priors')

subplot(1,2,2);
R_idx = 1:11;
cs    = c_map(round(linspace(60,230,length(R_idx))),:);
for r = 1:length(R_idx)
    tmp = squeeze(p_a(:,:,R_idx(r))); 

    plotCI(tmp',Apriors,'SEM',cs(r,:),cs(r,:),'over'); hold on;
    plot(Apriors,mean(tmp,1),'Color',cs(r,:), 'LineWidth',2); hold on;
end
ylim([-0.1 1.1]); title(sprintf('w%d', w0-1));
xlabel('gen A priors'); ylabel('Inference A'); 
title('Influence of R priors')

% influence on R
figure; 
subplot(1,2,1);
W_idx = 1:11;
for w = 1:length(W_idx)
    tmp = squeeze(p_r(W_idx(w),:,:)); 

    plotCI(tmp',Rpriors,'SEM',cs(w,:),cs(w,:),'over'); hold on;
    plot(Rpriors,mean(tmp,1),'Color',cs(w,:), 'LineWidth',2); hold on;
end
ylim([-0.1 1.1]); title(sprintf('w%d', w0-1));
xlabel('gen R priors'); ylabel('Inference R'); 
title('Influence of W priors')

subplot(1,2,2);
A_idx = 1:11;
for a = 1:length(A_idx)
    tmp = squeeze(p_r(:,A_idx(a),:)); 

    plotCI(tmp',Rpriors,'SEM',cs(a,:),cs(a,:),'over'); hold on;
    plot(Rpriors,mean(tmp,1),'Color',cs(a,:), 'LineWidth',2); hold on;
end
ylim([-0.1 1.1]); title(sprintf('w%d', w0-1));
xlabel('gen R priors'); ylabel('Inference R'); 
title('Influence of A priors')


