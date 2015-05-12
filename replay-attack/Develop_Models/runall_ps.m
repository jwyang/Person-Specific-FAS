Feat_Types = [1 3];
Methods = {'CS' 'OLS' 'PLS'};
%{
for t = 1:length(Feat_Types)
    for m = 1:length(Methods)
        devel_PSModels_specific_quality(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feat_Types(t), Methods{m});
    end
end
%}
%{
for m = 1:length(Methods)
    devel_PSModels_fusion_specific_quality(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feat_Types, Methods{m});
end
%}

for t = 1:length(Feat_Types)
    for m = 1:length(Methods)
        devel_FASModels_specific_quality(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feat_Types(t), Methods{m});
    end
end

%{
for t = 1:length(Feat_Types)
    for m = 1:length(Methods)
        devel_PSModels_withSubNum_specific_quality(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feat_Types(t), Methods{m});
    end
end
%}
%{
for m = 1:length(Methods)
    devel_PSModels_withSubNum_fusion_specific_quality(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Types, Quality_Type, 'L', Methods{m});
end
%}

