Feat_Names = {'MsLBP' 'HOG'};

Methods_Generic = {'NULL' 'PCA' 'CS' 'OLS' 'PLS'};
Methods_PS = {'PCA' 'CS' 'OLS' 'PLS'};

Perfs_Generic = cell(3, 5);
Perfs_PS = cell(3, 4);

SubIDs_train  = Labels_train_SL.SubID_train;
SubIDs_devel   = Labels_devel_SL.SubID_devel;
SubIDs_test   = Labels_test_SL.SubID_test;
SubIDs = [SubIDs_train, SubIDs_devel, SubIDs_test];

clientID = unique(SubIDs);
clientID_source  = unique(SubIDs_train);
clientID_target = unique([SubIDs_devel, SubIDs_test]);

[~, ~, ind_S] = intersect(clientID_source, clientID);
[~, ~, ind_T] = intersect(clientID_target, clientID);

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_Generic)
        Perfs_Generic{t, m} = load(strcat('GenericPerf_test_withSubNum_rep_', Feat_Names{t}, '_', Methods_Generic{m}, '_Enroll.mat'));
    end
end

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_PS)
        Perfs_PS{t, m} = load(strcat('PerSpecPerf_test_withSubNum_rep_', Feat_Names{t}, '_', Methods_PS{m}, '_Enroll.mat'));
    end
end

markertypes = {'o', 's', 'd'};
load colors;
figure;
for t = 1:length(Feat_Names)
    subplot(2, 1, t);
    hold on;
    % plot HTER curves for Generic models
    for m = 1:length(Methods_Generic)
        num_targets = length(Perfs_Generic{t, m}.Perf);
        hters = zeros(1, num_targets);
        for i = 1:num_targets
            hters(i) = mean(Perfs_Generic{t, m}.Perf{i}.HTERs(ind_T));
        end
        
        p(m) = plot(hters, 'Color', colors(m, :), 'LineWidth', 1.5, 'Marker', markertypes{t}, ...
            'MarkerSize', 4, 'MarkerEdgeColor', 'auto', 'MarkerFaceColor', 'none');
    end
    
    % plot ROC curves for Specific models
    for m = 1:length(Methods_PS)
        num_targets = length(Perfs_PS{t, m}.Perf);
        hters = zeros(1, num_targets);
        for i = 1:num_targets
            hters(i) = mean(Perfs_PS{t, m}.Perf{i}.HTERs(ind_T));
        end
        
        p(m + 5) = plot(hters, 'Color', colors(m + 5, :), 'LineWidth', 1.5, 'Marker', markertypes{t}, ...
            'MarkerSize', 4, 'MarkerEdgeColor', 'auto', 'MarkerFaceColor', 'none');
    end
    
    legend_generic = strcat('G-FAS,', {' '}, Feat_Names{t}, {', '}, Methods_Generic);
    legend_specific = strcat('PS-iFAS,', {' '}, Feat_Names{t}, {', '}, Methods_PS);
    legend(p(1:5), legend_generic, 'Color', 'none');
    % legend boxoff;
    ah=axes('position',get(gca,'position'),...
        'visible','off');
    legend(ah, p(6:9), legend_specific, 'Color', 'none');
    % legend([legend_generic, legend_specific]);
    % columnlegend(3, [legend_generic, legend_specific], 'Location', 'NorthWest');
end
%{
function hters = Process_Generic(Perfs, ind)
   num_targets = length(Perfs.Perf);
   hters = zeros(1, num_targets);
   for i = 1:num_targets
       hters(i) = mean(Perfs.Perf{i}.HTERs(ind));
   end
end

function hters = Process_Specific(Perfs,ind)
   num_targets = length(Perfs.Perf);
   hters = zeros(1, num_targets);
   for i = 1:num_targets
       hters(i) = mean(Perfs.Perf{i}.HTERs(ind));
   end
end
%}