Feat_Types = [3];
Methods = {'CS' 'OLS' 'PLS'};

for t = 1:length(Feat_Types)
    for m = 1:3
        TargetDA_AllQualities(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Types(t), methods{m});
    end
end