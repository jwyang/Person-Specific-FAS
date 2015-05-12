function plot_HTERs_withSubNum

Feat_Names = {'MsLBP' 'HOG'};

Methods_Generic = {'NULL' 'PCA' 'CS' 'OLS' 'PLS'};
Methods_PS = {'PCA' 'CS' 'OLS' 'PLS'};

Perfs_Generic = cell(3, 5);
Perfs_PS = cell(3, 4);

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_Generic)
        Perfs_Generic{t, m} = load(strcat('GenericPerf_test_withSubNum_rep_', Feat_Names{t}, '_', Methods_Generic{m}, '_M_L.mat'));
    end
end

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_PS)
        Perfs_PS{t, m} = load(strcat('PerSpecPerf_test_withSubNum_rep_', Feat_Names{t}, '_', Methods_PS{m}, '_M_L.mat'));
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
        hters = Process_Generic(Perfs_Generic{t, m});
        p(m) = plot(hters, 'Color', colors(m, :), 'LineWidth', 1.5, 'Marker', markertypes{t}, ...
            'MarkerSize', 4, 'MarkerEdgeColor', 'auto', 'MarkerFaceColor', 'none');
    end
    
    % plot ROC curves for Specific models
    for m = 1:length(Methods_PS)
        hters = Process_Specific(Perfs_PS{t, m});
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
end

function hters = Process_Generic(Perfs)
   num_targets = length(Perfs.Perf);
   hters = zeros(1, num_targets);
   for i = 1:num_targets
       hters(i) = mean(Perfs.Perf{i}.HTERs(21:50));
   end
end

function hters = Process_Specific(Perfs)
   num_targets = length(Perfs.Perf);
   hters = zeros(1, num_targets);
   for i = 1:num_targets
       hters(i) = mean(Perfs.Perf{i}.HTERs(21:50));
   end
end