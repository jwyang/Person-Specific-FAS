Feat_Names = {'MsLBP' 'HOG'};

Methods_Generic = {'NULL' 'PCA' 'CS' 'OLS' 'PLS'};
Methods_PS = {'PCA' 'CS' 'OLS' 'PLS'};
Methods_FAS = {'CS' 'OLS' 'PLS'};

Perfs_Generic = cell(3, 5);
Perfs_PS = cell(3, 4);
Perfs_FAS = cell(2, 3);

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

for t = 1:length(Feat_Names)
    for m = 1:length(Methods_FAS)
        Perfs_FAS{t, m} = load(strcat('FASPerf_test_', Feat_Names{t}, '_', Methods_FAS{m}, '_M_L.mat'));
    end
end

markertypes = {'o', 's', 'd'};
%{
for t = 1:length(Feat_Names) 
    % plot HTERs for Generic models
    figure;
    hold on;
    plot(Perfs_Generic{t, 1}.Perf.HTERs(1:20), 'Color', colors(1, :), 'LineWidth', 1.5, 'Marker', markertypes{t}, ...
       'MarkerSize', 4, 'MarkerEdgeColor', 'auto', 'MarkerFaceColor', 'none');    
   
    plot(Perfs_PS{t, 2}.Perf.HTERs(1:20), 'Color', colors(2 + 5, :), 'LineWidth', 1.5, 'Marker', markertypes{t}, ...
            'MarkerSize', 4, 'MarkerEdgeColor', 'auto', 'MarkerFaceColor', 'none');
end
%}
% plot HTERs
% Feat_Names{3} = 'Fusion';
load colors;
close all;
figure, hold on;
for t = 1:length(Feat_Names) 
    % plot HTERs for Generic models
    subplot(2, 1, t);
    hold on;
    
    stds_generic = cell(1, length(Methods_Generic));
    for m = 1:length(Methods_Generic)
        p(m) = plot(Perfs_Generic{t, m}.Perf.HTERs, 'Color', colors(m, :), 'LineWidth', 1.5, 'Marker', markertypes{t}, ...
            'MarkerSize', 4, 'MarkerEdgeColor', 'auto', 'MarkerFaceColor', 'none');
        stds_generic{m} = num2str(std(Perfs_Generic{t, m}.Perf.HTERs));
    end
    
    % plot HTERs for specific models
    stds_specfic = cell(1, length(Methods_PS));
    for m = 1:length(Methods_PS)
        p(m + 5) = plot(Perfs_PS{t, m}.Perf.HTERs, 'Color', colors(m + 5, :), 'LineWidth', 1.5, 'Marker', markertypes{t}, ...
            'MarkerSize', 4, 'MarkerEdgeColor', 'auto', 'MarkerFaceColor', 'none');
        stds_specfic{m} = num2str(std(Perfs_PS{t, m}.Perf.HTERs));
    end
    
    % plot HTERs for fused models
    stds_fused = cell(1, length(Methods_FAS));
    for m = 1:length(Methods_FAS)
        p(m + 9) = plot(Perfs_FAS{t, m}.Perf.HTERs, 'Color', colors(m + 5, :), 'LineWidth', 1.5, 'Marker', markertypes{t}, ...
            'MarkerSize', 4, 'MarkerEdgeColor', 'auto', 'MarkerFaceColor', 'none');
        stds_fused{m} = num2str(std(Perfs_FAS{t, m}.Perf.HTERs));
    end
    
    legend_generic = strcat('G-FAS,', {' '}, Feat_Names{t}, {', '}, Methods_Generic, {', '}, (stds_generic));
    legend_specific = strcat('PS-iFAS,', {' '}, Feat_Names{t}, {', '}, Methods_PS, {', '}, (stds_specfic));
    legend_fused = strcat('Fusion,', {' '}, Feat_Names{t}, {', '}, Methods_FAS, {', '}, (stds_fused));

    legend(p(1:5), legend_generic, 'Color', 'none');
    % legend boxoff;
    ah=axes('position',get(gca,'position'),...
        'visible','off');
    legend(ah, p(6:9), legend_specific, 'Color', 'none');    % legend boxoff;
    ah=axes('position',get(gca,'position'),...
        'visible','off');
    legend(ah, p(10:12), legend_fused, 'Color', 'none');
    % columnlegend(3, [legend_generic, legend_specific], 'Location', 'NorthWest', 'boxoff'); 
end