function draw_PSpecificHTERs(model, Feat_Type, method)
%DRAW_PSPECIFICHTERS Summary of this function goes here
%   Function: draw person-specific HTERs on test set
%   Detailed explanation goes here
%   model: Generic or Person-Specific
%   Feat_Type: feature type
%   method: domain adaptation method

name = {'MsLBP' 'LBP' 'HOG' 'LPQ'};

modelname = '';
for k = 1:length(Feat_Type)
    modelname = strcat(modelname, '_', name{Feat_Type(k)});
end

if strcmp(model, 'Generic')
    load(strcat('GenericPerf_test',modelname));    
    [~, ~, ind_S] = intersect(Perf.clientID_source, Perf.clientID);
    [~, ~, ind_T] = intersect(Perf.clientID_target, Perf.clientID);
    
    HTERs_S = Perf.HTERs(ind_S);  
    HTERs_T = Perf.HTERs(ind_T);
    
    HTERs = [HTERs_S; HTERs_T];
    
    plot(HTERs);  
    stddev = std(HTERs);  
    disp(strcat('Standard Deviation: ', num2str(stddev)));
    
elseif strcmp(model, 'PSpecific')
    load(strcat('PerSpecPerf_test', modelname, '_', method, '.mat'));
    
    [~, ~, ind_S] = intersect(Perf.clientID_source, Perf.clientID);
    [~, ~, ind_T] = intersect(Perf.clientID_target, Perf.clientID);
    
    HTERs_S = Perf.HTERs(ind_S);  
    HTERs_T = Perf.HTERs(ind_T);
    
    HTERs = [HTERs_S; HTERs_T];
    
    plot(HTERs);    
    stddev = std(HTERs);
    disp(strcat('Standard Deviation: ', num2str(stddev)));
    
end

end

