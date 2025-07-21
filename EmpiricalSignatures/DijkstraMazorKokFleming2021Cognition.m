clc;
clear all;
root = 'D:\PRM_HOSP\Matlab_code_vector_var';
load(fullfile(root,'Mental_imagery','simulation_imageryperception.mat'))

%% Cognition (2021) paper
m_idx = [1 7]; % no input vs threshold input
response = squeeze(p_aALL(m_idx,:,:));
pFA   = squeeze(mean(response(1,:,:),3));
pHit  = squeeze(mean(response(2,:,:),3)); 

figure; 
subplot(2,2,1); s = scatter([1 2]+(randn(nReps,2)./10),squeeze(response(1,:,:))',30,'filled');
s(1).MarkerFaceAlpha = 0.5; s(2).MarkerFaceAlpha = 0.5; 
hold on; boxplot(squeeze(response(1,:,:))'); ylim([0 1]); title('False alarms');

subplot(2,2,2);s = scatter([1 2]+(randn(nReps,2)./10),squeeze(response(2,:,:))',30,'filled');
s(1).MarkerFaceAlpha = 0.5; s(2).MarkerFaceAlpha = 0.5; title('Hits');
hold on; boxplot(squeeze(response(2,:,:))'); ylim([0 1]);

diff = squeeze(response(:,2,:)-response(:,1,:));
subplot(2,2,3); s = scatter(1+(randn(nReps,1)./10),squeeze(diff(1,:)),30,'filled');
s(1).MarkerFaceAlpha = 0.5; title('False alarms'); 
hold on; boxplot(squeeze(diff(1,:))'); ylim([-1 1]); hold on; plot(xlim,[0 0],'k--'); 

subplot(2,2,4); s = scatter(1+(randn(nReps,1)./10),squeeze(diff(2,:)),30,'filled');
s(1).MarkerFaceAlpha = 0.5; title('Hits'); 
hold on; boxplot(squeeze(diff(2,:))'); ylim([-1 1]); hold on; plot(xlim,[0 0],'k--');

% calculate criterion and d'
pHits = squeeze(response(2,:,:));
pFAs  = squeeze(response(1,:,:));
[dp, cr] = dprime(pHits,pFAs);
figure; subplot(1,4,1); b = barwitherr(std(dp,[],2)./sqrt(nReps),mean(dp,2)); title('d prime');
subplot(1,4,2); barwitherr(std(cr,[],2)./sqrt(nReps),mean(cr,2)); title('Criterion')
subplot(1,4,3); barwitherr(std(pFAs,[],2)./sqrt(nReps),mean(pFAs,2)); title('False alarms'); ylim([0 1]);
subplot(1,4,4); barwitherr(std(pHits,[],2)./sqrt(nReps),mean(pHits,2)); title('Hits'); ylim([0 1]);
