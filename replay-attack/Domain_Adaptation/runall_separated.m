
Feat_Types = [1 3];
Methods = {'CS' 'OLS' 'PLS' 'PCA'};
BTypes = {'AdvN' 'ContN'};

for b = 2
for t = 1
    for m = 1:3 % length(Methods)
        TargetDA_AllQualities(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feats_enroll_SL, Labels_enroll_SL, Feat_Types(t), BTypes{b}, Methods{m})        
    end
end
end
