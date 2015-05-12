% print EERs and HTERs for making tables
Feat_Names = {'MsLBP' 'HOG'};

Methods_Generic = {'NULL' 'PCA' 'CS' 'OLS' 'PLS'};
Methods_PS = {'PCA' 'CS' 'OLS' 'PLS'};

SubIDs_train  = Labels_train_SL.SubID_train;
SubIDs_devel   = Labels_devel_SL.SubID_devel;
SubIDs_test   = Labels_test_SL.SubID_test;
SubIDs = [SubIDs_train, SubIDs_devel, SubIDs_test];

clientID = unique(SubIDs);
clientID_source  = unique(SubIDs_train);
clientID_target = unique([SubIDs_devel, SubIDs_test]);

[~, ~, ind_S] = intersect(clientID_source, clientID);
[~, ~, ind_T] = intersect(clientID_target, clientID);
    
Perfs_Generic = cell(3, 5);
Perfs_PS = cell(3, 4);

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_Generic)
        Perfs_Generic{t, m} = load(strcat('../Develop_Models/GenericPerf_Sep_devel_', Feat_Names{t}, '_', Methods_Generic{m}, '_Enroll.mat'));
    end
end

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_PS)
        Perfs_PS{t, m} = load(strcat('../Develop_Models/PerSpecPerf_devel_Sep_', Feat_Names{t}, '_', Methods_PS{m}, '_Enroll.mat'));
    end
end

% display the EERs on development set in latex table format

for t = 1:length(Feat_Names)
    % line-1:EERs on source subjects
    disp_str = '';
    for m = 1:length(Methods_Generic)
        error = sprintf('%3.2f', 100 * mean(Perfs_Generic{t, m}.Perf.HTERs_persub(ind_S)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    
    for m = 1:length(Methods_PS)
        error = sprintf('%3.2f', 100 * mean(Perfs_PS{t, m}.Perf.EERs(ind_S)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    display(disp_str);
    % line-2:EERs on target subjects
    disp_str = '';
    for m = 1:length(Methods_Generic)
        error = sprintf('%3.2f', 100 * mean(Perfs_Generic{t, m}.Perf.HTERs_persub(ind_T)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    
    for m = 1:length(Methods_PS)
        error = sprintf('%3.2f', 100 * mean(Perfs_PS{t, m}.Perf.EERs(ind_T)));
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
        Perfs_Generic{t, m} = load(strcat('GenericPerf_Sep_test_', Feat_Names{t}, '_', Methods_Generic{m}, '_Enroll.mat'));
    end
end

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_PS)
        Perfs_PS{t, m} = load(strcat('PerSpecPerf_Sep_test_', Feat_Names{t}, '_', Methods_PS{m}, '_Enroll.mat'));
    end
end

% display the EERs on test set in latex table format
for t = 1:length(Feat_Names)
    % line-1:EERs on source subjects
    disp_str = '';
    for m = 1:length(Methods_Generic)
        error = sprintf('%3.2f', 100 * mean(Perfs_Generic{t, m}.Perf.HTERs(ind_S)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    
    for m = 1:length(Methods_PS)
        error = sprintf('%3.2f', 100 * mean(Perfs_PS{t, m}.Perf.HTERs(ind_S)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    display(disp_str);
    % line-2:EERs on target subjects
    disp_str = '';
    for m = 1:length(Methods_Generic)
        error = sprintf('%3.2f', 100 * mean(Perfs_Generic{t, m}.Perf.HTERs(ind_T)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    
    for m = 1:length(Methods_PS)
        error = sprintf('%3.2f', 100 * mean(Perfs_PS{t, m}.Perf.HTERs(ind_T)));
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
