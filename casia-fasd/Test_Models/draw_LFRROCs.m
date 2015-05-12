function draw_LFRROCs(model, Feat_Type, method)
%DRAW_LFRROCS Summary of this function goes here
%   Detailed explanation goes here

name = {'MsLBP' 'LBP' 'HOG' 'LPQ'};
modelname = '';
for k = 1:length(Feat_Type)
    modelname = strcat(modelname, '_', name{Feat_Type(k)});
end


if strcmp(model, 'Generic')
    % load(strcat('FRPerf_test', modelname, '.mat'));
    load(strcat('FRPerf_test', modelname, '_', 'CS', '.mat'));
    load(strcat('GenericPerf_test',modelname));
    
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
    
    thresholds = scores_min : step : scores_max;
    
    FARs = zeros(length(thresholds), 1);
    FRRs = zeros(length(thresholds), 1);
    SUB_NUM = length(Perf.HTERs);
    
    for t = 1:length(thresholds)
        
        FANum = zeros(SUB_NUM, 1);
        FRNum = zeros(SUB_NUM, 1);
        
        nums_gen = zeros(SUB_NUM, 1);
        nums_fake = zeros(SUB_NUM, 1);
        
        for s = 1:50
            %
            gen_num = length(Perf.Scores_gen{s});
            fake_num = length(Perf.Scores_fake{s});
            
            nums_gen(s) = gen_num;
            nums_fake(s) = fake_num;
            
            genLabels = FRLabels{s}.inds(1:gen_num);
            fakeLabels = FRLabels{s}.inds(gen_num+1:gen_num+fake_num);
            
            FANum(s) = sum(Perf.Scores_fake{s} >= (thresholds(t)));
            FRNum(s) = sum(Perf.Scores_gen{s} < (thresholds(t)) | genLabels ~= s); %
        end
        
        FARs(t) = sum(FANum)/sum(nums_fake);
        FRRs(t) = sum(FRNum)/sum(nums_gen);
        
    end
    
    plot(FARs, FRRs);
    
elseif strcmp(model, 'PSpecific')
    if matlabpool('size')<=0 %判断并行计算环境是否已然启动
        matlabpool('open','local',8); %若尚未启动，则启动并行环境
    else
        disp('Already initialized'); %说明并行环境已经启动。
    end
    
    % load('..\Develop_Models\PerSpecLFRPerf_devel_MsLBP_CS.mat');
    
    load(strcat('PerSpecPerf_crosstest', modelname, '_', method, '.mat'));
    load(strcat('FRPerf_CASIA_nofake_test_Generic', '.mat'));
    % load(strcat('FRPerf_test_HOG_CS.mat'));
    P_thresholds = Perf.thresholds;
    Scores_gen = Perf.Scores_gen;
    Scores_fake = Perf.Scores_fake;
    
    scores_genuine = [];
    scores_fake = [];
    
    for s = 1:length(Perf.HTERs)
        scores_genuine = [scores_genuine; Perf.Scores_gen{s}];
        scores_fake = [scores_fake; Perf.Scores_fake{s}];
    end
    
    scores_min = min([min(scores_genuine), min(scores_fake)]);
    scores_max = max([max(scores_genuine), max(scores_fake)]);
    
    scores_min = scores_min - (max(Perf.thresholds) - min(Perf.thresholds));
    scores_max = scores_max + (max(Perf.thresholds) - min(Perf.thresholds));
    
    Steps = 10000;
    
    step = (scores_max - scores_min)/Steps;
    
    thresholds = scores_min : step : scores_max;
    
    FARs = zeros(length(thresholds), 1);
    FRRs = zeros(length(thresholds), 1);
    
    SUB_NUM = length(Perf.thresholds);
    
    for s = 1:50
        gen_num = length(Scores_gen{s,s});
        fake_num = length(Scores_fake{s,s});
        
        n = length(FRLabels{s}.inds);
        gen_ind = (FRLabels{s}.gen_fake == 1);
        fake_ind = (FRLabels{s}.gen_fake == -1);
        FRLabels{s}.inds = [FRLabels{s}.inds(gen_ind); FRLabels{s}.inds(fake_ind)];
        
        if n < (gen_num + fake_num)
            FRLabels{s}.inds = [FRLabels{s}.inds; FRLabels{s}.inds(end-(gen_num + fake_num)+n+1:end)];
        end
    end
    
    parfor t = 1:length(thresholds)
        FANum = zeros(SUB_NUM, 1);
        FRNum = zeros(SUB_NUM, 1);
        
        nums_gen = zeros(SUB_NUM, 1);
        nums_fake = zeros(SUB_NUM, 1);
        
        for s = 1:50
            %
            gen_num = length(Scores_gen{s,s});
            fake_num = length(Scores_fake{s,s});
            
            nums_gen(s) = gen_num;
            nums_fake(s) = fake_num;
            
            genLabels = FRLabels{s}.inds(1:gen_num);
            fakeLabels = FRLabels{s}.inds(end-fake_num+1:end);
            
            % genLabels = FRLabels{s}.inds(1:gen_num);
            % fakeLabels = FRLabels{s}.inds(gen_num+1:gen_num+fake_num);
            
            FANum(s) = sum(Scores_fake{s, s} >= (P_thresholds(s) + thresholds(t)) & fakeLabels == s);
            
            % Add the number fake Samples which is incorrectly identified and recognized as genuine faces
            ind = find(fakeLabels ~= s);
            labels = zeros(length(ind), 1);
            for i = 1:length(ind)
                preSubID = fakeLabels(ind(i));
                Scores = Scores_fake{s, preSubID};
                labels(i) = Scores(ind(i)) >= (P_thresholds(preSubID) + thresholds(t));
            end
            FANum(s) = FANum(s) + sum(labels);
            
            FRNum(s) = sum(Scores_gen{s, s} < (P_thresholds(s) + thresholds(t)) | genLabels ~= s);    %
            
        end
        
        FARs(t) = sum(FANum)/sum(nums_fake);
        FRRs(t) = sum(FRNum)/sum(nums_gen);
        
    end
    
    plot(FARs, FRRs);

end

end

