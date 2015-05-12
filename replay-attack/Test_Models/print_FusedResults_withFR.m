% print EERs and HTERs for making tables
Feat_Names = {'MsLBP' 'HOG'};
Methods_PS = {'CS' 'OLS' 'PLS'};

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
        Perfs_FAS{t, m} = load(strcat('FASPerf_Sep_crosstest_', Feat_Names{t}, '_', Methods_FAS{m}, '_Enroll.mat'));
    end
end

load fr_results_subjects_genuine;
load fr_results_subjects_fake;

% display the EERs on test set in latex table format
for t = 1:length(Feat_Names)
    % line-1:EERs on source subjects
    disp_str = '';
    
    for m = 1:length(Methods_FAS)
        error = lfr_process(Perfs_FAS{t, m}.Perf, [fr_results_subjects_genuine; fr_results_subjects_fake], ind_S, clientID);
        error_disp = sprintf('%3.2f', 100 * error);
        disp_str = strcat(disp_str, '&', {' '}, error_disp, {' '});
    end
    display(disp_str);
    % line-2:EERs on target subjects
    disp_str = '';
    
    for m = 1:length(Methods_FAS)
        % error = sprintf('%3.2f', 100 * mean(Perfs_PS{t, m}.Perf.HTERs(ind_T)));
        error = lfr_process(Perfs_FAS{t, m}.Perf, [fr_results_subjects_genuine; fr_results_subjects_fake], ind_T, clientID);
        error_disp = sprintf('%3.2f', 100 * error);        
        disp_str = strcat(disp_str, '&', {' '}, error_disp, {' '});
    end
    display(disp_str);
    % line-3:EERs on all subjects
    disp_str = '';
    
    for m = 1:length(Methods_FAS)
        % error = sprintf('%3.2f', 100 * mean(Perfs_PS{t, m}.Perf.HTERs(1:end)));
        error = lfr_process(Perfs_FAS{t, m}.Perf, [fr_results_subjects_genuine; fr_results_subjects_fake], [1:50], clientID);
        error_disp = sprintf('%3.2f', 100 * error);        
        disp_str = strcat(disp_str, '&', {' '}, error_disp, {' '});
    end
    display(disp_str);
end
