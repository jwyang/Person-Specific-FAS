function TargetDA_AllQualities(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Type, method)
%TARGETDA_ALLQUALITIES Summary of this function goes here
%   Detailed explanation goes here

if matlabpool('size')<=0 %判断并行计算环境是否已然启动
    matlabpool('open','local',12); %若尚未启动，则启动并行环境
else
    disp('Already initialized'); %说明并行环境已经启动。
end

global Transform;
global Transform_CS;
name = {'MsLBP' 'LBP' 'HOG' 'LPQ'};

QualityTypes = {'HN' 'MN' 'LN'}

if ~strcmp(method, 'CS') && ~strcmp(method, 'PCA')
    T = load(strcat('..\Estimate_Transfomration_for_Expe\Transform_HP_HP_', name{Feat_Type}, '_', method, '.mat'));
    Transform = T.Transform;
    
    T = load(strcat('..\Estimate_Transfomration_for_Expe\Transform_HP_HP_', name{Feat_Type}, '_', 'CS', '.mat'));
    Transform_CS = T.Transform;
else
    T = load(strcat('..\Estimate_Transfomration_for_Expe\Transform_HP_HP_', name{Feat_Type}, '_', method, '.mat'));
    Transform = T.Transform;
end

for i = 1:3
    if ~strcmp(method, 'PCA')
        TargetDA(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Type, QualityTypes{i}, method);
        % TargetDAwithSubNum(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Type, QualityTypes{i}, method);
    else
        TargetDA_PCA_withSubNum(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Type, QualityTypes{i}, method);        
    end
end

end

