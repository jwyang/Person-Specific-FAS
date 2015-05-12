function plot_ROCs
Feat_Names = {'MsLBP' 'HOG' 'MsLBP_HOG'};

Methods_Generic = {'NULL' 'PCA' 'CS' 'OLS' 'PLS'};
Methods_PS = {'PCA' 'CS' 'OLS' 'PLS'};
Methods_FAS = {'CS' 'OLS' 'PLS'};

Perfs_Generic = cell(3, 5);
Perfs_PS = cell(3, 4);
Perfs_FAS = cell(2, 3);

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_Generic)
        Perfs_Generic{t, m} = load(strcat('GenericPerf_test_', Feat_Names{t}, '_', Methods_Generic{m}, '_M_L.mat'));
    end
end

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_PS)
        Perfs_PS{t, m} = load(strcat('PerSpecPerf_test_', Feat_Names{t}, '_', Methods_PS{m}, '_M_L.mat'));
    end
end

for t = 1:length(Feat_Names) - 1
    for m = 1:length(Methods_FAS)
        Perfs_FAS{t, m} = load(strcat('FASPerf_test_', Feat_Names{t}, '_', Methods_FAS{m}, '_M_L.mat'));
    end
end

% plot ROCs
Feat_Names{3} = 'Fusion';
load colors
% close all;
for t = 1:2 % length(Feat_Names)
    figure, hold on;
    % plot ROC curves for Generic models
    for m = 1:length(Methods_Generic)
        [FARs, FRRs] = Process_Generic(Perfs_Generic{t, m}.Perf.Scores_gen, Perfs_Generic{t, m}.Perf.Scores_fake);
        plot(FARs, FRRs, 'Color', colors(m, :), 'LineWidth', 1.5);
    end
    
    % plot ROC curves for Specific models
    for m = 1:length(Methods_PS)
        [FARs, FRRs] = Process_Specific(Perfs_PS{t, m}.Perf.Scores_gen, Perfs_PS{t, m}.Perf.Scores_fake, ...
            Perfs_PS{t, m}.Perf.thresholds);
        plot(FARs, FRRs, 'Color', colors(m, :), 'LineWidth', 1.5);
    end    
    
    % plot ROC curves for Fused models
    for m = 1:length(Methods_FAS)
        [FARs, FRRs] = Process_Specific(Perfs_FAS{t, m}.Perf.Scores_gen, Perfs_FAS{t, m}.Perf.Scores_fake, ...
            Perfs_FAS{t, m}.Perf.thresholds);
        plot(FARs, FRRs, 'Color', colors(m, :), 'LineWidth', 1.5);
    end    
    
    legend_generic = strcat('G-FAS,', {' '}, Feat_Names{t}, {', '}, Methods_Generic);
    legend_specific = strcat('PS-iFAS,', {' '}, Feat_Names{t}, {', '}, Methods_PS);
    legend_fused = strcat('PS-iFAS,', {' '}, Feat_Names{t}, {', '}, Methods_FAS);
    
    legend([legend_generic, legend_specific, legend_fused]);
end

function [FARs, FRRs] = Process_Generic(Scores_gen, Scores_fake)

scores_genuine = [];
scores_fake = [];

for s = 1:length(Scores_gen)
    scores_genuine = [scores_genuine; Scores_gen{s}];
    scores_fake = [scores_fake; Scores_fake{s}];    
end

scores_min = min([min(scores_genuine), min(scores_fake)]);
scores_max = max([max(scores_genuine), max(scores_fake)]);

Steps = 10000;

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

function [FARs, FRRs] = Process_Specific(Scores_gen, Scores_fake, thresholds)

scores_genuine = [];
scores_fake = [];

for s = 1:length(Scores_gen)
    scores_genuine = [scores_genuine; Scores_gen{s}];
    scores_fake = [scores_fake; Scores_fake{s}];    
end

scores_min = min([min(scores_genuine), min(scores_fake)]);
scores_max = max([max(scores_genuine), max(scores_fake)]);

Steps = 10000;
step = (scores_max - scores_min)/Steps;

threshold_min = min(thresholds);
threshold_max = max(thresholds);

scores_min = scores_min - (threshold_max - threshold_min);
scores_max = scores_max + (threshold_max - threshold_min);

thetas = scores_min : step : scores_max;

FARs = zeros(length(thetas), 1);
FRRs = zeros(length(thetas), 1);


num_genuine = length(scores_genuine);
num_fake = length(scores_fake);


for i = 1:length(thetas)
    FARNum = zeros(50, 1);
    FRRNum = zeros(50, 1);
    th = thetas(i);
    for s = 1:50
        FARNum(s) = sum(Scores_fake{s} >= (thresholds(s) + th));
        FRRNum(s) = sum(Scores_gen{s} < (thresholds(s) + th));        
    end
    
    FARs(i) = sum(FARNum)/num_fake;
    FRRs(i) = sum(FRRNum)/num_genuine;
end