% Replication of Lau HC, Passingham RE. (2006) Relative blindsight in 
% normal observers and the neural correlate of visual consciousness. PNAS
%
% Fig. 2: change in awareness without a change in discrimination
% performance
restoredefaultpath;
clc;
clear all;
root = 'D:\PRM_HOSS\Matlab_code_vector_var';
addpath(fullfile(root))
addpath(fullfile(root,'Utilities'))
addpath(fullfile(root,'Perception'))
cd(root)

%% Get from previous simulations
load(fullfile(root,'Perception','simulation_perception.mat'))

idx = [12 18];
discrimination = p_w1(idx,:);
detection      = p_a(idx,:);

figure;
subplot(1,2,1);
barwitherr(std(discrimination,[],2)./sqrt(10),mean(discrimination,2));
title('% Correct'); xlabel('Contrast'); ylim([0 1.2])

subplot(1,2,2);
barwitherr(std(detection,[],2)./sqrt(10),mean(detection,2));
title('% Seen'); xlabel('Contrast'); ylim([0 1.2])

