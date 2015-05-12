function [EER, threshold] = compute_EER(genuine_scores, fake_scores)
%COMPUTE_EER Summary of this function goes here
%   Function: compute the EER and the corrsponding threshold
%   Detailed explanation goes here
%   Input:
%       genuine_scores: scores from genuine samples
%       fake_scores: scores from fake smaples

scores_min = min([min(genuine_scores), min(fake_scores)]);
scores_max = max([max(genuine_scores), max(fake_scores)]);

Steps = 10000;

step = (scores_max - scores_min)/Steps;

thresholds = scores_min : step : scores_max;

FARs = zeros(length(thresholds), 1);
FRRs = zeros(length(thresholds), 1);


num_genuine = length(genuine_scores);
num_fake = length(fake_scores);

for s = 1:length(thresholds)
    FARs(s) = sum(fake_scores >= thresholds(s))/(num_fake);
    FRRs(s) = sum(genuine_scores < thresholds(s))/(num_genuine);
end

[~, ind] = min(abs(FARs - FRRs));
   
EER = (FARs(ind) + FRRs(ind))/2;
threshold = thresholds(ind);

end

