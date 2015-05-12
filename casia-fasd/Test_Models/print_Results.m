% print EERs and HTERs for making tables
Feat_Names = {'MsLBP' 'HOG'};

Methods_Generic = {'NULL' 'PCA' 'CS' 'OLS' 'PLS'};
Methods_PS = {'PCA' 'CS' 'OLS' 'PLS'};

Perfs_Generic = cell(3, 5);
Perfs_PS = cell(3, 4);

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_Generic)
        Perfs_Generic{t, m} = load(strcat('../Develop_Models/GenericPerf_devel_', Feat_Names{t}, '_', Methods_Generic{m}, '_M_L.mat'));
    end
end

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_PS)
        Perfs_PS{t, m} = load(strcat('../Develop_Models/PerSpecPerf_devel_', Feat_Names{t}, '_', Methods_PS{m}, '_M_L.mat'));
    end
end

% display the EERs on development set in latex table format

for t = 1:length(Feat_Names)
    % line-1:EERs on source subjects
    disp_str = '';
    for m = 1:length(Methods_Generic)
        error = sprintf('%3.2f', 100 * mean(Perfs_Generic{t, m}.Perf.HTERs_persub(1:20)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    
    for m = 1:length(Methods_PS)
        error = sprintf('%3.2f', 100 * mean(Perfs_PS{t, m}.Perf.EERs(1:20)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    display(disp_str);
    % line-2:EERs on target subjects
    disp_str = '';
    for m = 1:length(Methods_Generic)
        error = sprintf('%3.2f', 100 * mean(Perfs_Generic{t, m}.Perf.HTERs_persub(21:50)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    
    for m = 1:length(Methods_PS)
        error = sprintf('%3.2f', 100 * mean(Perfs_PS{t, m}.Perf.EERs(21:end)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    display(disp_str);
    % line-3:EERs on all subjects
    disp_str = '';
    for m = 1:length(Methods_Generic)
        error = sprintf('%3.2f', 100 * Perfs_Generic{t, m}.Perf.EER_overall);
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    
    for m = 1:length(Methods_PS)
        error = sprintf('%3.2f', 100 * mean(Perfs_PS{t, m}.Perf.EERs(1:end)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    display(disp_str);
end


for t = 1:length(Feat_Names)
    for m = 1:length(Methods_Generic)
        Perfs_Generic{t, m} = load(strcat('GenericPerf_test_', Feat_Names{t}, '_', Methods_Generic{m}, '_M_L.mat'));
    end
end

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_PS)
        Perfs_PS{t, m} = load(strcat('PerSpecPerf_test_', Feat_Names{t}, '_', Methods_PS{m}, '_M_L.mat'));
    end
end

% display the EERs on test set in latex table format
for t = 1:length(Feat_Names)
    % line-1:EERs on source subjects
    disp_str = '';
    for m = 1:length(Methods_Generic)
        error = sprintf('%3.2f', 100 * mean(Perfs_Generic{t, m}.Perf.HTERs(1:20)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    
    for m = 1:length(Methods_PS)
        error = sprintf('%3.2f', 100 * mean(Perfs_PS{t, m}.Perf.HTERs(1:20)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    display(disp_str);
    % line-2:EERs on target subjects
    disp_str = '';
    for m = 1:length(Methods_Generic)
        error = sprintf('%3.2f', 100 * mean(Perfs_Generic{t, m}.Perf.HTERs(21:50)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    
    for m = 1:length(Methods_PS)
        error = sprintf('%3.2f', 100 * mean(Perfs_PS{t, m}.Perf.HTERs(21:end)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    display(disp_str);
    % line-3:EERs on all subjects
    disp_str = '';
    for m = 1:length(Methods_Generic)
        error = sprintf('%3.2f', 100 * Perfs_Generic{t, m}.Perf.HTER);
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    
    for m = 1:length(Methods_PS)
        error = sprintf('%3.2f', 100 * mean(Perfs_PS{t, m}.Perf.HTERs(1:end)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    display(disp_str);
end
