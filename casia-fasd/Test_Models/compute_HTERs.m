function compute_HTERs( model, Feat_Type, method )
%COMPUTE_HTERS Summary of this function goes here
%   Detailed explanation goes here


name = {'MsLBP' 'LBP' 'HOG' 'LPQ'};
modelname = '';
for k = 1:length(Feat_Type)
    modelname = strcat(modelname, '_', name{Feat_Type(k)});
end


if strcmp(model, 'Generic')
    % load(strcat('FRPerf_test', modelname, '.mat'));
    % load(strcat('FRPerf_test_withSynth', modelname, '_', 'CS', '.mat'));
    load('FRPerf_CASIA_test_Generic.mat');
    
    load(strcat('GenericPerf_test',modelname));    
    SUB_NUM = length(Perf.HTERs);
    FANum = zeros(SUB_NUM, 1);
    FRNum = zeros(SUB_NUM, 1);
    
    nums_gen = zeros(SUB_NUM, 1);
    nums_fake = zeros(SUB_NUM, 1);
    
    for s = 21:50
        % 
        gen_num = length(Perf.Scores_gen{s});
        fake_num = length(Perf.Scores_fake{s});
        
        nums_gen(s) = gen_num;
        nums_fake(s) = fake_num;
        
        n = length(FRLabels{s}.inds); 
        
        
        gen_ind = find(FRLabels{s}.gen_fake == 1);
        fake_ind = find(FRLabels{s}.gen_fake == -1);
        FRLabels{s}.inds = [FRLabels{s}.inds(gen_ind); FRLabels{s}.inds(fake_ind)];
        
        
        if n < (gen_num + fake_num)            
            FRLabels{s}.inds = [FRLabels{s}.inds; FRLabels{s}.inds(end-(gen_num + fake_num)+n+1:end)];            
        end
        
        genLabels = FRLabels{s}.inds(1:gen_num);
        fakeLabels = FRLabels{s}.inds(end-fake_num+1:end);
        
        % genLabels = FRLabels{s}.inds(1:gen_num);
        % fakeLabels = FRLabels{s}.inds(gen_num+1:gen_num+fake_num);
        
        FANum(s) = sum(Perf.Scores_fake{s} >= (Perf.Threshold));
        FRNum(s) = sum(Perf.Scores_gen{s} < (Perf.Threshold) | genLabels ~= s); % 
    end
    
    FAR = sum(FANum)/sum(nums_fake);
    FRR = sum(FRNum)/sum(nums_gen);
    
    HTER = (FAR + FRR)/2;
    
    disp(strcat('HTER: ', num2str(HTER)));
elseif strcmp(model, 'PSpecific')
    
    % load('..\Develop_Models\PerSpecLFRPerf_devel_MsLBP_CS.mat');
    % thresholds = Perf.thresholds;
    
    load(strcat('PerSpecPerf_crosstest', modelname, '_', method, '.mat'));
    load(strcat('FRPerf_CASIA_nofake_test_Generic', '.mat'));
    % load(strcat('FRPerf_test_MsLBP_CS.mat'));
    thresholds = Perf.thresholds;

    % load(strcat('FRPerf_test_HOG_CS.mat'));
    
    SUB_NUM = length(Perf.thresholds);
    FANum = zeros(SUB_NUM, 1);
    FRNum = zeros(SUB_NUM, 1);
    
    nums_gen = zeros(SUB_NUM, 1);
    nums_fake = zeros(SUB_NUM, 1);
    FARs = zeros(SUB_NUM, 1);
    FRRs = zeros(SUB_NUM, 1);
    HTERs = zeros(SUB_NUM, 1);
    
    % mean(Perf.HTERs)
    nums_errorfr = zeros(SUB_NUM, 1);
    for s = 1:50
        % 
        gen_num = length(Perf.Scores_gen{s, s});
        fake_num = length(Perf.Scores_fake{s, s});
        
        nums_gen(s) = gen_num;
        nums_fake(s) = fake_num;
        
        n = length(FRLabels{s}.inds); 
        gen_ind = find(FRLabels{s}.gen_fake == 1);
        fake_ind = find(FRLabels{s}.gen_fake == -1);
        FRLabels{s}.inds = [FRLabels{s}.inds(gen_ind); FRLabels{s}.inds(fake_ind)];
        
        if n < (gen_num + fake_num)            
            FRLabels{s}.inds = [FRLabels{s}.inds; FRLabels{s}.inds(end-(gen_num + fake_num)+n+1:end)];            
        end
        
        genLabels = FRLabels{s}.inds(1:gen_num);
        fakeLabels = FRLabels{s}.inds(end-fake_num+1:end);
        nums_errorfr(s) = sum(FRLabels{s}.inds ~= s);
        
        FANum(s) = sum(Perf.Scores_fake{s, s} >= (thresholds(s)) & fakeLabels == s); % 
        
        % Add the number fake Samples which is incorrectly identified and recognized as genuine faces
        ind = find(fakeLabels ~= s);
        for i = 1:length(ind)
            FANum(s) = FANum(s) + (Perf.Scores_fake{s, fakeLabels(ind(i))}(ind(i)) >= thresholds(fakeLabels(ind(i))));
        end
        
        % FRNum(s) = sum(Perf.Scores_gen{s, s} < (thresholds(s)) | genLabels ~= s);    % 
        FRNum(s) = sum(Perf.Scores_gen{s, s} < (thresholds(s)) & genLabels == s);    % 
        ind = find(genLabels ~= s);
        for i = 1:length(ind)
            FRNum(s) = FRNum(s) + (Perf.Scores_gen{s, genLabels(ind(i))}(ind(i)) < thresholds(genLabels(ind(i))));
        end
        
        FARs(s) = (FANum(s))/(nums_fake(s));
        FRRs(s) = (FRNum(s))/(nums_gen(s));
        HTERs(s) = (FARs(s) + FRRs(s))/2;
    end
    
    FAR = sum(FANum)/sum(nums_fake);
    FRR = sum(FRNum)/sum(nums_gen);
    fraccuracy = 1-sum(nums_errorfr)/(sum(nums_gen)+sum(nums_fake));
    
    HTER = (FAR + FRR)/2;
    
    disp(strcat('HTER: ', num2str(HTER), 'FRAccuracy: ', num2str(fraccuracy)));
    
end

end

