function draw_PSpecificROCs(Feat_Type, method)
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

load(strcat('PerSpecPerf_test', modelname, '_', method, '.mat'));

scores_genuine = [];
scores_fake = [];

for s = 1:length(Perf.HTERs)
    scores_genuine = [scores_genuine; Perf.Scores_gen{s}];
    scores_fake = [scores_fake; Perf.Scores_fake{s}];    
end

scores_min = min([min(scores_genuine), min(scores_fake)]);
scores_max = max([max(scores_genuine), max(scores_fake)]);

Steps = 10000;
step = (scores_max - scores_min)/Steps;

threshold_min = min(Perf.thresholds);
threshold_max = max(Perf.thresholds);

scores_min = scores_min - (threshold_max - threshold_min);
scores_max = scores_max + (threshold_max - threshold_min);

thresholds = scores_min : step : scores_max;

FARs = zeros(length(thresholds), 1);
FRRs = zeros(length(thresholds), 1);


num_genuine = length(scores_genuine);
num_fake = length(scores_fake);


for i = 1:length(thresholds)
    FARNum = zeros(50, 1);
    FRRNum = zeros(50, 1);
    th = thresholds(i);
    for s = 1:50
        FARNum(s) = sum(Perf.Scores_fake{s} >= (Perf.thresholds(s) + th));
        FRRNum(s) = sum(Perf.Scores_gen{s} < (Perf.thresholds(s) + th));        
    end
    
    FARs(i) = sum(FARNum)/num_fake;
    FRRs(i) = sum(FRRNum)/num_genuine;
end

plot(FARs, FRRs);

end

