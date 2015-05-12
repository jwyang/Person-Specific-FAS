function train_FRModel_fusion(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Type)
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
method_name = {'CS' 'LR' 'PLS'};

SubIDs_train  = Labels_train_SL.SubID_train;
QLabels_train = Labels_train_SL.QLabels_train;
TLabels_train = Labels_train_SL.TLabels_train;
Feats_train   = Feats_train_SL.Feats_train;

SubIDs_test   = Labels_test_SL.SubID_test;
QLabels_test  = Labels_test_SL.QLabels_test;
TLabels_test  = Labels_test_SL.TLabels_test;
Feats_test    = Feats_test_SL.Feats_test;

SubIDs_test   = SubIDs_test + 20;

% concatenate train and test information
SubIDs = [SubIDs_train, SubIDs_test];
QLabels = [QLabels_train, QLabels_test];
TLabels = [TLabels_train, TLabels_test];
Feats   = [Feats_train, Feats_test];

SUB_NUM_S = 20;
SUB_NUM_T = 30;
SUB_NUM = SUB_NUM_S + SUB_NUM_T;

samples_label_subject = zeros(length(SubIDs), 1);
samples_label_subjects = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    samples_label_subjects{s} = uint8(samples_label_subject);
end

% strategy 2: use only genuine faces to train face recognition classfiers
% strategy 1: use both genuine faces and fake faces to train face recognition classfiers

% strategy 2
for i = 1:length(SubIDs)
    s = SubIDs(i);
    if strcmp(TLabels(i), 'PP')
        samples_label_subjects{s}(i) = 1;
    end
end


global samples_data_subjects;
global samples_stype_subjects_genuine;
global samples_qtype_subjects_genuine;

samples_data_subjects = cell(1, SUB_NUM);
samples_stype_subjects_genuine = cell(1, SUB_NUM);
samples_qtype_subjects_genuine = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    samples_data_subjects{s} = zeros(sum(samples_label_subjects{s}), sum(dims(Feat_Type)));
    
    samples_stype_subjects_genuine{s} = cell(sum(samples_label_subjects{s}), 1);
    samples_qtype_subjects_genuine{s} = cell(sum(samples_label_subjects{s}), 1);
end

dims_feat = [0 cumsum(dims(Feat_Type))];

sample_sub_id = ones(1, SUB_NUM);

for i = 1:length(SubIDs)
    s = SubIDs(i);
    if strcmp(TLabels(i), 'PP')
        for k = 1:length(Feat_Type)
        if Feat_Type(k) == 1            
            samples_data_subjects{s}(sample_sub_id(s), dims_feat(k)+1:dims_feat(k+1)) = Feats(i).MsLBP{1}/norm(Feats(i).MsLBP{1});
        elseif Feat_Type(k) == 2
            samples_data_subjects{s}(sample_sub_id(s), dims_feat(k)+1:dims_feat(k+1)) = Feats(i).LBP{1}/norm(Feats(i).LBP{1});
        elseif Feat_Type(k) == 3
            samples_data_subjects{s}(sample_sub_id(s), dims_feat(k)+1:dims_feat(k+1)) = Feats(i).HOG{1}/norm(Feats(i).HOG{1});
        elseif Feat_Type(k) == 4
            samples_data_subjects{s}(sample_sub_id(s), dims_feat(k)+1:dims_feat(k+1)) = Feats(i).LPQ{1}/norm(Feats(i).LPQ{1});
        end
        end
        samples_stype_subjects_genuine{s}(sample_sub_id(s)) = TLabels(i);
        samples_qtype_subjects_genuine{s}(sample_sub_id(s)) = QLabels(i);     
        sample_sub_id(s) = sample_sub_id(s) + 1;
    end
end

samples_data_allsubjects = [];
samples_label_allsubjects = [];

QualityTypes_Genuine = {'LP' 'MP' 'HP'};

for s = 1:SUB_NUM
    % reassign genuine samples
    samples_gen_data_sub = [];
    for q = 1:3
        ind = find(strcmp(samples_qtype_subjects_genuine{s}, QualityTypes_Genuine{q}));
        samples_gen_data_sub = [samples_gen_data_sub; samples_data_subjects{s}(ind(1:int16(length(ind)/3)), :)];
    end
    samples_data_subjects{s} = samples_gen_data_sub;
    samples_data_allsubjects = [samples_data_allsubjects; samples_gen_data_sub];
    samples_label_allsubjects = [samples_label_allsubjects; s*ones(size(samples_gen_data_sub, 1), 1)];
end
% normalize feature vectors
% samples_data_allsubjects = bsxfun(@rdivide, samples_data_allsubjects, sqrt(sum(samples_data_allsubjects.^2, 2)));

method = 'directlda';
[A,T] = directlda(samples_data_allsubjects,samples_label_allsubjects,SUB_NUM-1,method);

samples_transform_data_subject = cell(1, SUB_NUM);

for i = 1:SUB_NUM
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

save(strcat('FRModel',  modelname, '.mat'), 'FRModel');

end



