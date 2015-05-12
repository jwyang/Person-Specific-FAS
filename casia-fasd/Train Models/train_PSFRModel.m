function train_PSFRModel(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Type, method)
%TRAIN_FRMODEL Summary of this function goes here
%   Function: train face recognition model
%   Detailed explanation goes here
%   Input:
%        Feats_train_SL: feature set of first 20 subjects
%        Labels_train_SL: label set of first 20 subjects
%        Feats_test_SL: feature set of remaining 30 subjects
%        Labels_test_SL: feature set of remaning 30 subjects
%        Feat_Type: type of feature used for face anti-spoofing
%        method: the method for estimating transformation, the methods are
%        (1) Center-Shift, (2) OLS, (3) PLS

dims = [833 361 378 256];
name = {'MsLBP' 'LBP' 'HOG' 'LPQ'};

SUB_NUM_S = 20;
SUB_NUM_T = 30;
SUB_NUM = SUB_NUM_S + SUB_NUM_T;

SubIDs_train  = Labels_train_SL.SubID_train;
TLabels_train = Labels_train_SL.TLabels_train;
QLabels_train = Labels_train_SL.QLabels_train;
Feats_train   = Feats_train_SL.Feats_train;

SubIDs_test   = Labels_test_SL.SubID_test;
TLabels_test  = Labels_test_SL.TLabels_test;
QLabels_test = Labels_test_SL.QLabels_test;
Feats_test    = Feats_test_SL.Feats_test;

SubIDs_test   = SubIDs_test + 20;

% concatenate train and test information
SubIDs = [SubIDs_train, SubIDs_test];
TLabels = [TLabels_train, TLabels_test];
QLabels = [QLabels_train, QLabels_test];
Feats   = [Feats_train, Feats_test];

% initialize 
samples_label_subject_genuine = zeros(length(SubIDs), 1);
samples_label_subjects_genuine = cell(1, SUB_NUM);

samples_label_subject_fake = zeros(length(SubIDs), 1);
samples_label_subjects_fake = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    samples_label_subjects_genuine{s} = uint8(samples_label_subject_genuine);
    samples_label_subjects_fake{s} = uint8(samples_label_subject_fake);    
end

for i = 1:length(SubIDs)
    s = SubIDs(i);
    if s <= SUB_NUM_S
        if strcmp(TLabels(i), 'PP')
            samples_label_subjects_genuine{s}(i) = 1;
        else
            samples_label_subjects_fake{s}(i) = 1;            
        end
    elseif s > SUB_NUM_S && s < SUB_NUM
        if strcmp(TLabels(i), 'PP')
            samples_label_subjects_genuine{s}(i) = 1;            
        end
    end
end

global samples_data_subjects_genuine;
global samples_data_subjects_fake;

global samples_stype_subjects_genuine;
global samples_qtype_subjects_genuine;

global samples_stype_subjects_fake;
global samples_qtype_subjects_fake;

samples_data_subjects_genuine = cell(1, SUB_NUM);
samples_data_subjects_fake = cell(1, SUB_NUM);

samples_stype_subjects_genuine = cell(1, SUB_NUM);
samples_stype_subjects_fake = cell(1, SUB_NUM);

samples_qtype_subjects_genuine = cell(1, SUB_NUM);
samples_qtype_subjects_fake = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    samples_data_subjects_genuine{s} = zeros(sum(samples_label_subjects_genuine{s}), sum(dims(Feat_Type)));
    samples_data_subjects_fake{s} = zeros(sum(samples_label_subjects_fake{s}), sum(dims(Feat_Type)));    
    
    samples_stype_subjects_genuine{s} = cell(sum(samples_label_subjects_fake{s}), 1);
    samples_stype_subjects_fake{s} = cell(sum(samples_label_subjects_fake{s}), 1);

    samples_qtype_subjects_genuine{s} = cell(sum(samples_label_subjects_fake{s}), 1);
    samples_qtype_subjects_fake{s} = cell(sum(samples_label_subjects_fake{s}), 1);    
end

dims_feat = [0 cumsum(dims(Feat_Type))];

sample_sub_id_genuine = ones(1, SUB_NUM);
sample_sub_id_fake = ones(1, SUB_NUM);

for i = 1:length(SubIDs)
    s = SubIDs(i);
    if s <= 20
        if strcmp(TLabels(i), 'PP')
            for k = 1:length(Feat_Type)
            if Feat_Type(k) == 1
                samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), dims_feat(k)+1:dims_feat(k+1)) = Feats(i).MsLBP{1}/norm(Feats(i).MsLBP{1});
            elseif Feat_Type(k) == 2
                samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), dims_feat(k)+1:dims_feat(k+1)) = Feats(i).LBP{1}/norm(Feats(i).LBP{1});
            elseif Feat_Type(k) == 3
                samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), dims_feat(k)+1:dims_feat(k+1)) = Feats(i).HOG{1}/norm(Feats(i).HOG{1});
            elseif Feat_Type(k) == 4
                samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), dims_feat(k)+1:dims_feat(k+1)) = Feats(i).LPQ{1}/norm(Feats(i).LPQ{1});
            end
            end
            samples_stype_subjects_genuine{s}(sample_sub_id_genuine(s)) = TLabels(i);
            samples_qtype_subjects_genuine{s}(sample_sub_id_genuine(s)) = QLabels(i);            
            sample_sub_id_genuine(s) = sample_sub_id_genuine(s) + 1;
        else
            for k = 1:length(Feat_Type)
            if Feat_Type(k) == 1
                samples_data_subjects_fake{s}(sample_sub_id_fake(s), dims_feat(k)+1:dims_feat(k+1)) = Feats(i).MsLBP{1}/norm(Feats(i).MsLBP{1});
            elseif Feat_Type(k) == 2
                samples_data_subjects_fake{s}(sample_sub_id_fake(s), dims_feat(k)+1:dims_feat(k+1)) = Feats(i).LBP{1}/norm(Feats(i).LBP{1});                
            elseif Feat_Type(k) == 3
                samples_data_subjects_fake{s}(sample_sub_id_fake(s), dims_feat(k)+1:dims_feat(k+1)) = Feats(i).HOG{1}/norm(Feats(i).HOG{1});
            elseif Feat_Type(k) == 4
                samples_data_subjects_fake{s}(sample_sub_id_fake(s), dims_feat(k)+1:dims_feat(k+1)) = Feats(i).LPQ{1}/norm(Feats(i).LPQ{1});
            end
            end
            samples_stype_subjects_fake{s}(sample_sub_id_fake(s)) = TLabels(i);
            samples_qtype_subjects_fake{s}(sample_sub_id_fake(s)) = QLabels(i);    
            sample_sub_id_fake(s) = sample_sub_id_fake(s) + 1;
        end
    elseif s > 20 && s <= SUB_NUM
        if strcmp(TLabels(i), 'PP')
            for k = 1:length(Feat_Type)
            if Feat_Type(k) == 1
                samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), dims_feat(k)+1:dims_feat(k+1)) = Feats(i).MsLBP{1}/norm(Feats(i).MsLBP{1});
            elseif Feat_Type(k) == 2
                samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), dims_feat(k)+1:dims_feat(k+1)) = Feats(i).LBP{1}/norm(Feats(i).LBP{1});
            elseif Feat_Type(k) == 3
                samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), dims_feat(k)+1:dims_feat(k+1)) = Feats(i).HOG{1}/norm(Feats(i).HOG{1});
            elseif Feat_Type(k) == 4
                samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), dims_feat(k)+1:dims_feat(k+1)) = Feats(i).LPQ{1}/norm(Feats(i).LPQ{1});
            end
            end
            samples_stype_subjects_genuine{s}(sample_sub_id_genuine(s)) = TLabels(i);
            samples_qtype_subjects_genuine{s}(sample_sub_id_genuine(s)) = QLabels(i);   
            sample_sub_id_genuine(s) = sample_sub_id_genuine(s) + 1;
        end
    end
end

% Use only the fisrt 1/3 part of samples for training person-specific face anti-spoofing classifier
QualityTypes_Genuine = {'LP' 'MP' 'HP'};
QualityTypes_Fake = {'LN' 'MN' 'HN'};
SpoofingTypes = {'CPN' 'WPN' 'VN'};

for s = 1:SUB_NUM
    % reassign genuine samples
    samples_gen_data_sub = [];
    for q = 1:3
        ind = find(strcmp(samples_qtype_subjects_genuine{s}, QualityTypes_Genuine{q}));
        samples_gen_data_sub = [samples_gen_data_sub; samples_data_subjects_genuine{s}(ind(1:int16(length(ind)/3)), :)];
    end
    samples_data_subjects_genuine{s} = samples_gen_data_sub;
    
    if s > SUB_NUM_S
        continue;
    end
    
    % reassign fake samples
    samples_fake_data_sub = [];    
    for a = 1:3
        for q = 1:3        
            ind = find(strcmp(samples_stype_subjects_fake{s}, SpoofingTypes{a}) & strcmp(samples_qtype_subjects_fake{s}, QualityTypes_Fake{q}));
            samples_fake_data_sub = [samples_fake_data_sub; samples_data_subjects_fake{s}(ind(1:int16(length(ind)/3)), :)];
        end
    end
    samples_data_subjects_fake{s} = samples_fake_data_sub;
end


% Load virtual fake features of target subjects for training
VirtualFeatures = cell(3, length(Feat_Type));

for q = 1:3
    for k = 1:length(Feat_Type)
        VirtualFeatures{q, k} = load(strcat('..\Domain_Adaptation\SynthFeatures_AllTSubs_', QualityTypes_Fake{q}, '_', name{Feat_Type(k)}, '_', method));
    end
end

% T = load(strcat('..\Estimate_Transfomration_for_Expe\Transform_HN_LN_', name{Feat_Type}, '_CS.mat'));
% Transform_HN_LN = T.Transform_CS;
% 
% T = load(strcat('..\Estimate_Transfomration_for_Expe\Transform_HN_MN_', name{Feat_Type}, '_CS.mat'));
% Transform_HN_MN = T.Transform_CS;
% 
% CS_HN_LN = mean(Transform_HN_LN.CenterShiftVectors(1:SUB_NUM_S, :), 1);
% CS_HN_MN = mean(Transform_HN_MN.CenterShiftVectors(1:SUB_NUM_S, :), 1);

% for s = SUB_NUM_S+1:SUB_NUM
% 
%     samples_data_q1 = [];
%     for k = 1:length(Feat_Type)
%         data = VirtualFeatures{1, k}.SynthFeatures_AllTSubs{s-SUB_NUM_S};
%         if isempty(data)
%             break;
%         end
%         data = bsxfun(@rdivide, data, sqrt(sum(data.^2, 2)));  
%         if k == 1
%             samples_data_q1 = data;
%             continue;
%         end
%         if size(samples_data_q1, 1) > size(data, 1)     
%         samples_data_q1 = [samples_data_q1(1:size(data, 1), :), data];
%         elseif size(samples_data_q1, 1) <= size(data, 1)
%         samples_data_q1 = [samples_data_q1, data(1:size(samples_data_q1, 1), :)];            
%         end
%     end
%     
%     samples_data_q2 = [];
%     for k = 1:length(Feat_Type)
%         data = VirtualFeatures{2, k}.SynthFeatures_AllTSubs{s-SUB_NUM_S};
%         if isempty(data)
%             break;
%         end
%         data = bsxfun(@rdivide, data, sqrt(sum(data.^2, 2)));     
%         if k == 1
%             samples_data_q2 = data;
%             continue;
%         end
%         if size(samples_data_q2, 1) > size(data, 1)     
%         samples_data_q2 = [samples_data_q2(1:size(data, 1), :), data];
%         elseif size(samples_data_q2, 1) <= size(data, 1)
%         samples_data_q2 = [samples_data_q2, data(1:size(samples_data_q2, 1), :)];            
%         end
%     end    
%     % Translate high quality samples according to the distributions of low and normal quality samples
% %     center_lq = mean(VirtualFeatures{1}.SynthFeatures_AllTSubs{s-SUB_NUM_S}, 1);
% %     center_mq = mean(VirtualFeatures{2}.SynthFeatures_AllTSubs{s-SUB_NUM_S}, 1);
% %     
% %     center_hq_lq = center_lq + CS_HN_LN;
% %     center_hq_mq = center_mq + CS_HN_MN;
% %     
% %     center_hq = (center_hq_lq + center_hq_mq)/2;    
% %     center_hq_f = mean(VirtualFeatures{3}.SynthFeatures_AllTSubs{s-SUB_NUM_S}, 1);
% %     
% %     VirtualFeatures{3}.SynthFeatures_AllTSubs{s-SUB_NUM_S} = bsxfun(@plus, VirtualFeatures{3}.SynthFeatures_AllTSubs{s-SUB_NUM_S}, (center_hq - center_hq_f));
%     samples_data_q3 = [];
%     for k = 1:length(Feat_Type)
%         data = VirtualFeatures{3, k}.SynthFeatures_AllTSubs{s-SUB_NUM_S};
%         if isempty(data)
%             break;
%         end
%         data = bsxfun(@rdivide, data, sqrt(sum(data.^2, 2)));      
%         if k == 1
%             samples_data_q3 = data;
%             continue;
%         end
%         if size(samples_data_q3, 1) > size(data, 1)
%             samples_data_q3 = [samples_data_q3(1:size(data, 1), :), data];
%         elseif size(samples_data_q3, 1) <= size(data, 1)
%             samples_data_q3 = [samples_data_q3, data(1:size(samples_data_q3, 1), :)];
%         end
%     end
% %     
% %     samples_data(samples_data < 0) = 0;
%     
%     samples_data_subjects_fake{s} = [samples_data_q1; samples_data_q2; samples_data_q3];
% end


samples_data_allsubjects = [];
samples_label_allsubjects = [];
for s = 1:SUB_NUM
    % reassign genuine samples
    samples_gen_data_sub = samples_data_subjects_genuine{s};
    samples_fake_data_sub = samples_data_subjects_fake{s};
    
    samples_data_sub = [samples_gen_data_sub; samples_fake_data_sub];
    samples_data_allsubjects = [samples_data_allsubjects; samples_data_sub];
    samples_label_allsubjects = [samples_label_allsubjects; s*ones(size(samples_data_sub, 1), 1)];
end
% normalize feature vectors
% samples_data_allsubjects = bsxfun(@rdivide, samples_data_allsubjects, sqrt(sum(samples_data_allsubjects.^2, 2)));

method_lda = 'pcalda';
[A,T] = directlda(samples_data_allsubjects,samples_label_allsubjects,SUB_NUM-1,method_lda);

samples_transform_data_subject = cell(1, SUB_NUM);

parfor i = 1:SUB_NUM
    samples_transform_data_subject{i} = T*samples_data_allsubjects(samples_label_allsubjects == i, :)';
end

FRModel.A = A;
FRModel.T = T;
FRModel.samples_transform = samples_transform_data_subject;
FRModel.method = method;
FRModel.dim = SUB_NUM-1;

modelname = '';
for k = 1:length(Feat_Type)
    modelname = strcat(modelname, '_', name{Feat_Type(k)});
end

save(strcat('PSFRModel_noSynth_',  modelname, '_', method, '.mat'), 'FRModel');

end



