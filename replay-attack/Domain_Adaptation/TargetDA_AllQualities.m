function TargetDA_AllQualities(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feats_enroll_SL, Labels_enroll_SL, Feat_Type, BLabel, method)
%TARGETDA_ALLQUALITIES Summary of this function goes here
%   Detailed explanation goes here

%{
if matlabpool('size')<=0 %判断并行计算环境是否已然启动
    matlabpool('open','local',8); %若尚未启动，则启动并行环境
else
    disp('Already initialized'); %说明并行环境已经启动。
end
%}

global Transform;
global Transform_CS;

name = {'MsLBP' 'LBP' 'HOG' 'LPQ'};

if ~isempty(BLabel)
    BLabel_str = strcat(BLabel, '_');
end

if strcmp(method, 'CS')
    T = load(strcat('..\Estimate_Transformationn_for_Expe\Transform_Enroll_', BLabel_str, name{Feat_Type}, '_', method, '.mat'));
    Transform = T.Transform;
elseif strcmp(method, 'OLS') || strcmp(method, 'PLS')
    T = load(strcat('..\Estimate_Transformationn_for_Expe\Transform_Enroll_', BLabel_str, name{Feat_Type}, '_', method, '.mat'));
    Transform = T.Transform;
    
    T = load(strcat('..\Estimate_Transformationn_for_Expe\Transform_Enroll_', BLabel_str, name{Feat_Type}, '_', 'CS', '.mat'));
    Transform_CS = T.Transform;
elseif strcmp(method, 'PCA')
    T = load(strcat('..\Estimate_Transformationn_for_Expe\Transform_withSubNum_', BLabel_str, name{Feat_Type}, '_', method, '.mat'));
    Transform = T.Transform;
end

clear T;

if ~strcmp(method, 'PCA')
    % TargetDA(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feats_enroll_SL, Labels_enroll_SL, Feat_Type, BLabel, method);    
    % TargetDA_combined(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feats_enroll_SL, Labels_enroll_SL, Feat_Type, method);
    TargetDA_withSubNum(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feats_enroll_SL, Labels_enroll_SL, Feat_Type, BLabel, method);
    % TargetDA_withSubNum_combined(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feats_enroll_SL, Labels_enroll_SL, Feat_Type, method);
else
    % TargetDA_PCA(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feats_enroll_SL, Labels_enroll_SL, Feat_Type, BLabel, method);    
    TargetDA_PCA_withSubNum(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feats_enroll_SL, Labels_enroll_SL, Feat_Type, BLabel, method);    
end

end

