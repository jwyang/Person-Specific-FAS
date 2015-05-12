% print EERs and HTERs for making tables
Feat_Names = {'MsLBP' 'HOG' 'MsLBP_HOG'};
Methods_PS = {'PCA' 'CS' 'OLS' 'PLS'};

SubIDs_train  = Labels_train_SL.SubID_train;
SubIDs_test   = Labels_test_SL.SubID_test;
SubIDs = [SubIDs_train, SubIDs_test];

clientID = unique(SubIDs);
clientID_source  = unique(SubIDs_train);
clientID_target = unique(SubIDs_test);
    
Perfs_PS = cell(3, 4);

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_PS)
        Perfs_PS{t, m} = load(strcat('PerSpecPerf_crosstest_', Feat_Names{t}, '_', Methods_PS{m}, '.mat'));
    end
end

load fr_results_subjects_genuine;
load fr_results_subjects_fake;

% display the EERs on test set in latex table format
for t = 1:length(Feat_Names)
    % line-1:EERs on source subjects
    disp_str = '';
    
    for m = 1:length(Methods_PS)
        error = lfr_process(Perfs_PS{t, m}.Perf, [fr_results_subjects_genuine; fr_results_subjects_fake], 1:20, 1:50);
        error_disp = sprintf('%3.2f', 100 * error);
        disp_str = strcat(disp_str, '&', {' '}, error_disp, {' '});
    end
    display(disp_str);
    % line-2:EERs on target subjects
    disp_str = '';
    
    for m = 1:length(Methods_PS)
        % error = sprintf('%3.2f', 100 * mean(Perfs_PS{t, m}.Perf.HTERs(ind_T)));
        error = lfr_process(Perfs_PS{t, m}.Perf, [fr_results_subjects_genuine; fr_results_subjects_fake], 21:50, 1:50);
        error_disp = sprintf('%3.2f', 100 * error);        
        disp_str = strcat(disp_str, '&', {' '}, error_disp, {' '});
    end
    display(disp_str);
    % line-3:EERs on all subjects
    disp_str = '';
    
    for m = 1:length(Methods_PS)
        % error = sprintf('%3.2f', 100 * mean(Perfs_PS{t, m}.Perf.HTERs(1:end)));
        error = lfr_process(Perfs_PS{t, m}.Perf, [fr_results_subjects_genuine; fr_results_subjects_fake], 1:50, 1:50);
        error_disp = sprintf('%3.2f', 100 * error);        
        disp_str = strcat(disp_str, '&', {' '}, error_disp, {' '});
    end
    display(disp_str);
end
