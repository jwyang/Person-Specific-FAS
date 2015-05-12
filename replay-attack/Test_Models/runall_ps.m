Feat_Types = [1 3];
Methods = {'CS' 'OLS' 'PLS' 'PCA'};
Quality_Type = {'M'};
%{
for t = 1:1 % length(Feat_Types)
    for m = 1:2 % length(Methods)
        test_PSModels_specific_quality(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feat_Types(t), Methods{m});
    end
end
%}


%{
% for t = 1:length(Feat_Types)
for m = 1:length(Methods)
    test_PSModels_fusion_specific_quality(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feat_Types, Methods{m});
end
% end
%}
%{
for t = 1:length(Feat_Types)
    for m = 4:length(Methods)
        test_PSModels_withSubNum_specific_quality(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feat_Types(t), Methods{m});
    end
end
%}
%{
for m = 1:length(Methods)
    test_PSModels_withSubNum_fusion_specific_quality(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Types, Quality_Type, 'L', Methods{m});
end
%}
%{
for t = 2:length(Feat_Types)
    for m = 1:length(Methods)
        test_PSLFRModels_specific_quality(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feat_Types(t), Methods{m});
    end
end
%}

for t = 2:length(Feat_Types)
    for m = 1:length(Methods)
        test_FASModels_specific_quality(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feat_Types(t), Methods{m});
    end
end

%{
for m = 1:length(Methods)
    test_PSLFRModels_specific_quality(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feat_Types, Methods{m});
end
%}