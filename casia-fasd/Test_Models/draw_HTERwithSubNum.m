function draw_HTERwithSubNum(model, Feat_Type, method)
%DRAW_HTERWITHSUBNUM Summary of this function goes here
%   Function: draw HTER curves with the increase of source subject number
%   Detailed explanation goes here
%   Input:
%      model: generic or person-specific
%      Feat_Type: MsLBP or HOG
%      method: generic, CS, OLS, PLS

name = {'MsLBP' 'LBP' 'HOG' 'LPQ'};
modelname = '';
for k = 1:length(Feat_Type)
    modelname = strcat(modelname, '_', name{Feat_Type(k)});
end

if strcmp(model, 'Generic')
    HTERs_T = zeros(20, 1);
    load(strcat('GenericPerf_test_withSubNum', modelname, '.mat'));

    for SubNum = 1:20
        SUB_NUM = 50;
        FANum = zeros(SUB_NUM, 1);
        FRNum = zeros(SUB_NUM, 1);
        
        nums_gen = zeros(SUB_NUM, 1);
        nums_fake = zeros(SUB_NUM, 1);
        
        for s = 21:50
            %
            gen_num = length(Perf{SubNum}.Scores_gen{s});
            fake_num = length(Perf{SubNum}.Scores_fake{s});
            
            nums_gen(s) = gen_num;
            nums_fake(s) = fake_num;
            
            FANum(s) = sum(Perf{SubNum}.Scores_fake{s} >= (Perf{SubNum}.Threshold));
            
            % Add the number fake Samples which is incorrectly identified and recognized as genuine faces
            
            FRNum(s) = sum(Perf{SubNum}.Scores_gen{s} < (Perf{SubNum}.Threshold));    %
            
        end
        
        FAR = sum(FANum)/sum(nums_fake);
        FRR = sum(FRNum)/sum(nums_gen);
        HTERs_T(SubNum) = (FAR + FRR)/2;
    end
    plot(HTERs_T);
    
elseif strcmp(model, 'PSpecific')
    load(strcat('PerSpecPerf_test_withSubNum', modelname, '_', method, '.mat'));
    HTERs_T = zeros(20, 1);
    for SubNum = 1:20
        
        SUB_NUM = 50;
        FANum = zeros(SUB_NUM, 1);
        FRNum = zeros(SUB_NUM, 1);
        
        nums_gen = zeros(SUB_NUM, 1);
        nums_fake = zeros(SUB_NUM, 1);
        FARs = zeros(SUB_NUM, 1);
        FRRs = zeros(SUB_NUM, 1);
        HTERs = zeros(SUB_NUM, 1);
        
        for s = 21:50
            %
            gen_num = length(Perf{SubNum}.Scores_gen{s});
            fake_num = length(Perf{SubNum}.Scores_fake{s});
            
            nums_gen(s) = gen_num;
            nums_fake(s) = fake_num;
            
            FANum(s) = sum(Perf{SubNum}.Scores_fake{s} >= (Perf{SubNum}.thresholds(s)));
            
            % Add the number fake Samples which is incorrectly identified and recognized as genuine faces
            
            FRNum(s) = sum(Perf{SubNum}.Scores_gen{s} < (Perf{SubNum}.thresholds(s)));    %
            
        end
        
        FAR = sum(FANum)/sum(nums_fake);
        FRR = sum(FRNum)/sum(nums_gen);
        HTERs_T(SubNum) = (FAR + FRR)/2;
    end
    
    plot(HTERs_T);
end

end

