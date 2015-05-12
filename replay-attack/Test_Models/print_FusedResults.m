% print EERs and HTERs for making tables
Feat_Names = {'MsLBP' 'HOG'};
Methods_FAS = {'CS' 'OLS' 'PLS'};

SubIDs_train  = Labels_train_SL.SubID_train;
SubIDs_devel   = Labels_devel_SL.SubID_devel;
SubIDs_test   = Labels_test_SL.SubID_test;
SubIDs = [SubIDs_train, SubIDs_devel, SubIDs_test];

clientID = unique(SubIDs);
clientID_source  = unique(SubIDs_train);
clientID_target = unique([SubIDs_devel, SubIDs_test]);

[~, ~, ind_S] = intersect(clientID_source, clientID);
[~, ~, ind_T] = intersect(clientID_target, clientID);

Perfs_FAS = cell(2, 3);

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_FAS)
        Perfs_FAS{t, m} = load(strcat('../Develop_Models/FASPerf_devel_Sep_', Feat_Names{t}, '_', Methods_FAS{m}, '_Enroll.mat'));
    end
end

% display the EERs on development set in latex table format

for t = 1:length(Feat_Names)
    % line-1:EERs on source subjects
    disp_str = '';
    
    for m = 1:length(Methods_FAS)
        error = sprintf('%3.2f', 100 * mean(Perfs_FAS{t, m}.Perf.EERs(ind_S)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    display(disp_str);
    % line-2:EERs on target subjects
    disp_str = '';
    
    for m = 1:length(Methods_FAS)
        error = sprintf('%3.2f', 100 * mean(Perfs_FAS{t, m}.Perf.EERs(ind_T)));
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
        Perfs_FAS{t, m} = load(strcat('FASPerf_Sep_test_', Feat_Names{t}, '_', Methods_FAS{m}, '_Enroll.mat'));
    end
end


% display the EERs on test set in latex table format
for t = 1:length(Feat_Names)
    % line-1:EERs on source subjects
    disp_str = '';
    
    for m = 1:length(Methods_FAS)
        error = sprintf('%3.2f', 100 * mean(Perfs_FAS{t, m}.Perf.HTERs(ind_S)));
        disp_str = strcat(disp_str, '&', {' '}, error, {' '});
    end
    display(disp_str);
    % line-2:EERs on target subjects
    disp_str = '';
    
    for m = 1:length(Methods_FAS)
        error = sprintf('%3.2f', 100 * mean(Perfs_FAS{t, m}.Perf.HTERs(ind_T)));
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
