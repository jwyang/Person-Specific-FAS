%{
if matlabpool('size')<=0 %判断并行计算环境是否已然启动
    matlabpool('open','local',4); %若尚未启动，则启动并行环境
else
    disp('Already initialized'); %说明并行环境已经启动。
end
%}
Feat_Types = [1 3];
BLabels = {'ContN' 'AdvN'};

for t = 1:length(Feat_Types)
for b = 1:length(BLabels)
  % EstimateTransformation_Enroll_Iterative(Feats_enroll_SL, Labels_train_SL, Labels_devel_SL, Labels_test_SL, Labels_enroll_SL, BLabels{b}, Feat_Types(t), 'CS');
  % EstimateTransformation_Enroll_Iterative(Feats_enroll_SL, Labels_train_SL, Labels_devel_SL, Labels_test_SL, Labels_enroll_SL, BLabels{b}, Feat_Types(t), 'OLS');
  % EstimateTransformation_Enroll_Iterative(Feats_enroll_SL, Labels_train_SL, Labels_devel_SL, Labels_test_SL, Labels_enroll_SL, BLabels{b}, Feat_Types(t), 'PLS');
  % EstimateTransformation_Enroll_PCA(Feats_enroll_SL, Labels_train_SL, Labels_devel_SL, Labels_test_SL, Labels_enroll_SL, Feat_Types(t), BLabels{b}, 'PCA');  
  EstimateTransformation_Enroll_PCA_withSubNum(Feats_enroll_SL, Labels_train_SL, Labels_devel_SL, Labels_test_SL, Labels_enroll_SL, Feat_Types(t), BLabels{b}, 'PCA');
end
end

