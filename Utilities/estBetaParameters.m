function [params] = estBetaParameters(mu,var)

params(1) = ((1 - mu) / var - 1 / mu) * mu ^ 2;
params(2)  = params(1) * (1 / mu - 1);