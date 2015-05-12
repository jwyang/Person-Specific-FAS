if matlabpool('size')<=0 %判断并行计算环境是否已然启动
    matlabpool('open','local',8); %若尚未启动，则启动并行环境
else
    disp('Already initialized'); %说明并行环境已经启动。
end

Feat_Types = [1 3];
Methods = {'CS' 'OLS' 'PLS' 'PCA'};

%{
for m = 4:4 % length(Methods)
    for t = 1:length(Feat_Types)
        test_PSModels_withSubNum( Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Types(t), Methods{m});
        % test_PSLFRModel(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Types(t), Methods{m});
    end
   test_PSModels_withSubNum_fusion(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Types, Methods{m});
   % test_PSLFRModel(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Types, Methods{m});
end
%}

figure, hold on;
for m = 1:length(Methods)
    for t = 1:length(Feat_Types)
        draw_LFRROCs('PSpecific', Feat_Types(t), Methods{m});        
    end
    draw_LFRROCs('PSpecific', Feat_Types, Methods{m});  
end


% for t = 1:length(Feat_Types)
%     test_GenericModel(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Types(t));
%     for m = 1:length(Methods)
%         test_PSModels(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Types(t), Methods{m});
%     end
% end
% 
% test_GenericModel_withSubNum_fusion(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Types);
% 
% for m = 1:length(Methods)
%     test_PSModels_fusion(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Types, Methods{m});
% end