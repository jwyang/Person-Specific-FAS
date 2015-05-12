function draw_GenericROCs(Feat_Type)
%DRAWROCS Summary of this function goes here
%   Function: Draw roc curves for generic model
%   Detailed explanation goes here
%   Input:
%      Feat_Type: MsLBP, HOG, or their combination

name = {'MsLBP' 'LBP' 'HOG' 'LPQ'};

modelname = '';
for k = 1:length(Feat_Type)
    modelname = strcat(modelname, '_', name{Feat_Type(k)});
end

modelname = strcat(modelname, '.mat');

load(strcat('GenericPerf_test',modelname));

scores_genuine = [];
scores_fake = [];

for s = 1:length(Perf.HTERs)
    scores_genuine = [scores_genuine; Perf.Scores_gen{s}];
    scores_fake = [scores_fake; Perf.Scores_fake{s}];    
end

scores_min = min([min(scores_genuine), min(scores_fake)]);
scores_max = max([max(scores_genuine), max(scores_fake)]);

Steps = 5000;

step = (scores_max - scores_min)/Steps;

thresholds = scores_min : step : scores_max;

FARs = zeros(length(thresholds), 1);
FRRs = zeros(length(thresholds), 1);


num_genuine = length(scores_genuine);
num_fake = length(scores_fake);

for s = 1:length(thresholds)
    FARs(s) = sum(scores_fake >= thresholds(s))/(num_fake);
    FRRs(s) = sum(scores_genuine < thresholds(s))/(num_genuine);
end

plot(FARs, FRRs);

end

