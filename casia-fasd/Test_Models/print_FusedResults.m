% print EERs and HTERs for making tables
Feat_Names = {'MsLBP' 'HOG'};

Methods_FAS = {'CS' 'OLS' 'PLS'};

Perfs_FAS = cell(3, 4);

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_FAS)
        Perfs_FAS{t, m} = load(strcat('../Develop_Models/FASPerf_devel_', Feat_Names{t}, '_', Methods_FAS{m}, '_M_L.mat'));
    end
end

% display the EERs on development set in latex table format

for t = 1:length(Feat_Names)
    % line-1:EERs on source subjects
    disp_str = '';
    
    for m = 1:length(Methods_FAS)
        error = sprintf('%3.2f', 100 * mean(Perfs_FAS{t, m}.Perf.EERs(1:20)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    display(disp_str);
    % line-2:EERs on target subjects
    disp_str = '';
    
    for m = 1:length(Methods_FAS)
        error = sprintf('%3.2f', 100 * mean(Perfs_FAS{t, m}.Perf.EERs(21:end)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    display(disp_str);
    % line-3:EERs on all subjects
    disp_str = '';
    
    for m = 1:length(Methods_FAS)
        error = sprintf('%3.2f', 100 * mean(Perfs_FAS{t, m}.Perf.EERs(1:end)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    display(disp_str);
end

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_FAS)
        Perfs_FAS{t, m} = load(strcat('FASPerf_test_', Feat_Names{t}, '_', Methods_FAS{m}, '_M_L.mat'));
    end
end

% display the EERs on test set in latex table format
for t = 1:length(Feat_Names)
    % line-1:EERs on source subjects
    disp_str = '';

    for m = 1:length(Methods_FAS)
        error = sprintf('%3.2f', 100 * mean(Perfs_FAS{t, m}.Perf.HTERs(1:20)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    display(disp_str);
    % line-2:EERs on target subjects
    disp_str = '';
    
    for m = 1:length(Methods_FAS)
        error = sprintf('%3.2f', 100 * mean(Perfs_FAS{t, m}.Perf.HTERs(21:end)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    display(disp_str);
    % line-3:EERs on all subjects
    disp_str = '';
    
    for m = 1:length(Methods_FAS)
        error = sprintf('%3.2f', 100 * mean(Perfs_FAS{t, m}.Perf.HTERs(1:end)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    display(disp_str);
end
