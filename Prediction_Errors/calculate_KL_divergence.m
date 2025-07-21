function [KL, prior, post] = calculate_KL_divergence(prior,posterior)

nDims = size(prior,2);

if nDims == 1

    % binning
    nBins = 100;
    edges = linspace(0, 1, nBins+1);  % Bin edges from 0 to 1
    prior = histcounts(prior,edges,'Normalization','probability');
    post = histcounts(posterior,edges,'Normalization','probability');

    % to prevent zeroes
    epsilon = 1e-10;
    post = post + epsilon;
    prior = prior + epsilon;

    % renormalize
    post = post / sum(post);
    prior = prior / sum(prior);

    % calculate KL divergence
    KL = sum(post.*(log(post./prior)));

elseif nDims == 2

    % binning
    nBins = 30;
    edges = linspace(-3, 3, nBins+1);
    prior = histcounts2(prior(:,1), prior(:,2), edges, edges);   
    post  = histcounts2(posterior(:,1),posterior(:,2),edges, edges);

    % to prevent zeroes
    epsilon = 1e-10;
    post = post + epsilon;
    prior = prior + epsilon;

    % renormalize
    post = post / sum(post(:));
    prior = prior / sum(prior(:));

    % calculate KL divergence
    KL = sum(post(:).*(log(post(:)./prior(:))));

end