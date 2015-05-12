Feat_Names = {'MsLBP' 'HOG'};

Methods_Generic = {'NULL' 'PCA' 'CS' 'OLS' 'PLS'};
Methods_PS = {'PCA' 'CS' 'OLS' 'PLS'};

Perfs_Generic = cell(3, 5);
Perfs_PS = cell(3, 4);

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_Generic)
        Perfs_Generic{t, m} = load(strcat('GenericPerf_Sep_test_', Feat_Names{t}, '_', Methods_Generic{m}, '_Enroll.mat'));
    end
end

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_PS)
        Perfs_PS{t, m} = load(strcat('PerSpecPerf_Sep_test_', Feat_Names{t}, '_', Methods_PS{m}, '_Enroll.mat'));
    end
end

Perfs_Generic_withSubNum = cell(3, 5);
Perfs_PS_withSubNum = cell(3, 4);

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_Generic)
        load(strcat('GenericPerf_test_withSubNum_', Feat_Names{t}, '_', Methods_Generic{m}, '_Enroll.mat'));
        Perf{15} = Perfs_Generic{t, m}.Perf;
        save(strcat('GenericPerf_test_withSubNum_rep_', Feat_Names{t}, '_', Methods_Generic{m}, '_Enroll.mat'), 'Perf');
    end
end

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_PS)
        load(strcat('PerSpecPerf_test_withSubNum_', Feat_Names{t}, '_', Methods_PS{m}, '.mat'));
        Perf{15} = Perfs_PS{t, m}.Perf;
        save(strcat('PerSpecPerf_test_withSubNum_rep_', Feat_Names{t}, '_', Methods_PS{m}, '_Enroll.mat'), 'Perf');
    end
end

