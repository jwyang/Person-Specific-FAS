function train_GenericModel_fusion_specific_quality(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, ...
    Feats_enroll_SL, Labels_enroll_SL, Feat_Type, method)
%TRAIN_GENERICMODEL Summary of this function goes here
%   Detailed explanation goes here
%   Input:
%        Feats_train_SL: feature set of first 20 subjects
%        Labels_train_SL: label set of first 20 subjects
%        Feats_test_SL: feature set of remaining 30 subjects
%        Labels_test_SL: feature set of remaning 30 subjects
%        Feat_Type: type of feature used for face anti-spoofing

% Step 1: Organize genuine training samples for all 20+30 subjects (first 1/3 part of all genuien samples)
name = {'MsLBP' 'LBP' 'HOG' 'LPQ'};
dims = [833 361 378 256];

SubIDs_train  = Labels_train_SL.SubID_train;
PNLabels_train = Labels_train_SL.PNLabels_train;
BLabels_train = Labels_train_SL.BLabels_train;
MLabels_train = Labels_train_SL.MLabels_train;
ALabels_train = Labels_train_SL.ALabels_train;
FLabels_train = Labels_train_SL.FLabels_train;
Feats_train   = Feats_train_SL.Feats_train;

SubIDs_devel   = Labels_devel_SL.SubID_devel;
PNLabels_devel  = Labels_devel_SL.PNLabels_devel;
BLabels_devel  = Labels_devel_SL.BLabels_devel;
MLabels_devel  = Labels_devel_SL.MLabels_devel;
ALabels_devel = Labels_devel_SL.ALabels_devel;
FLabels_devel = Labels_devel_SL.FLabels_devel;
Feats_devel    = Feats_devel_SL.Feats_devel;

SubIDs_test   = Labels_test_SL.SubID_test;
PNLabels_test  = Labels_test_SL.PNLabels_test;
BLabels_test  = Labels_test_SL.BLabels_test;
MLabels_test  = Labels_test_SL.MLabels_test;
ALabels_test = Labels_test_SL.ALabels_test;
FLabels_test = Labels_test_SL.FLabels_test;
Feats_test    = Feats_test_SL.Feats_test;

SubIDs_enroll   = Labels_enroll_SL.SubID_enroll;
PNLabels_enroll  = Labels_enroll_SL.PNLabels_enroll;
BLabels_enroll  = Labels_enroll_SL.BLabels_enroll;
MLabels_enroll  = Labels_enroll_SL.MLabels_enroll;
ALabels_enroll = Labels_enroll_SL.ALabels_enroll;
FLabels_enroll = Labels_enroll_SL.FLabels_enroll;
Feats_enroll    = Feats_enroll_SL.Feats_enroll;

% concatenate train and test information
SubIDs = [SubIDs_train, SubIDs_devel, SubIDs_test];
PNLabels = [PNLabels_train, PNLabels_devel, PNLabels_test];
BLabels = [BLabels_train, BLabels_devel, BLabels_test];
MLabels = [MLabels_train, MLabels_devel, MLabels_test];
ALabels = [ALabels_train, ALabels_devel, ALabels_test];
FLabels = [FLabels_train, FLabels_devel, FLabels_test];
Feats   = [Feats_train, Feats_devel, Feats_test];

clientID = unique(SubIDs);
clientID_enroll = unique(SubIDs_enroll);

clientID_source  = unique(SubIDs_train);
clientID_target = unique([SubIDs_devel, SubIDs_test]);

SUB_NUM_S = length(clientID_source);
SUB_NUM_T = length(clientID_target);
SUB_NUM = SUB_NUM_S + SUB_NUM_T;

samples_label_subject = zeros(length(SubIDs), 1);

samples_label_subjects_genuine = cell(1, SUB_NUM);
samples_label_subjects_fake = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    samples_label_subjects_genuine{s} = uint8(zeros(length(SubIDs_enroll), 1));
    samples_label_subjects_fake{s} = uint8(zeros(length(SubIDs), 1));    
end

for i = 1:length(SubIDs)
    s_rank = find(SubIDs(i)==clientID);
    s = s_rank(1);
    if strcmp(MLabels(i), 'FixN') && sum(SubIDs(i) == clientID_source) > 0
        samples_label_subjects_fake{s}(i) = 1;
    end
end

for i = 1:length(SubIDs_enroll)
    s_rank = find(SubIDs_enroll(i)==clientID_enroll);
    s = s_rank(1);
    samples_label_subjects_genuine{s}(i) = 1;
end

global samples_data_subjects_genuine;
global samples_data_subjects_fake;

samples_data_subjects_genuine = cell(1, SUB_NUM);         % save real and virtual fake samples for training
samples_data_subjects_fake = cell(1, SUB_NUM);     % save genuine samples for training

samples_btype_subjects_genuine = cell(1, SUB_NUM);

samples_btype_subjects_fake = cell(1, SUB_NUM);
samples_mtype_subjects_fake= cell(1, SUB_NUM);
samples_atype_subjects_fake = cell(1, SUB_NUM);
samples_ftype_subjects_fake = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    samples_data_subjects_genuine{s} = zeros(sum(samples_label_subjects_genuine{s}), sum(dims(Feat_Type)));
    samples_data_subjects_fake{s} = zeros(sum(samples_label_subjects_fake{s}), sum(dims(Feat_Type)));    
    
    samples_btype_subjects_genuine{s} = cell(sum(samples_label_subjects_genuine{s}), 1);
    
    samples_btype_subjects_fake{s} = cell(sum(samples_label_subjects_fake{s}), 1);
    samples_mtype_subjects_fake{s} = cell(sum(samples_label_subjects_fake{s}), 1);
    samples_atype_subjects_fake{s} = cell(sum(samples_label_subjects_fake{s}), 1);
    samples_ftype_subjects_fake{s} = cell(sum(samples_label_subjects_fake{s}), 1);
    
end

dims_feat = [0 cumsum(dims(Feat_Type))];
sample_sub_id_genuine = ones(1, SUB_NUM);
sample_sub_id_fake = ones(1, SUB_NUM);

for i = 1:length(SubIDs)
    s_rank = find(SubIDs(i)==clientID);
    s = s_rank(1);
    if strcmp(MLabels(i), 'FixN') && sum(SubIDs(i) == clientID_source) > 0
        for k = 1:length(Feat_Type)
        if Feat_Type(k) == 1
            samples_data_subjects_fake{s}(sample_sub_id_fake(s), dims_feat(k)+1:dims_feat(k+1)) = Feats{i}.MsLBP{1}/norm(Feats{i}.MsLBP{1});
        elseif Feat_Type(k) == 2
            samples_data_subjects_fake{s}(sample_sub_id_fake(s), dims_feat(k)+1:dims_feat(k+1)) = Feats{i}.LBP{1}/norm(Feats{i}.LBP{1});
        elseif Feat_Type(k) == 3
            samples_data_subjects_fake{s}(sample_sub_id_fake(s), dims_feat(k)+1:dims_feat(k+1)) = Feats{i}.HOG{1}/norm(Feats{i}.HOG{1});
        elseif Feat_Type(k) == 4
            samples_data_subjects_fake{s}(sample_sub_id_fake(s), dims_feat(k)+1:dims_feat(k+1)) = Feats{i}.LPQ{1}/norm(Feats{i}.LPQ{1});
        end
        end
        samples_mtype_subjects_fake{s}(sample_sub_id_fake(s)) = MLabels(i);
        samples_atype_subjects_fake{s}(sample_sub_id_fake(s)) = ALabels(i);
        samples_ftype_subjects_fake{s}(sample_sub_id_fake(s)) = FLabels(i);
        sample_sub_id_fake(s) = sample_sub_id_fake(s) + 1;
    end
end

for i = 1:length(SubIDs_enroll)
    s_rank = find(SubIDs_enroll(i)==clientID_enroll);
    s = s_rank(1);
        for k = 1:length(Feat_Type)
        if Feat_Type(k) == 1
            samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), dims_feat(k)+1:dims_feat(k+1)) = Feats_enroll{i}.MsLBP{1}/norm(Feats_enroll{i}.MsLBP{1});
        elseif Feat_Type(k) == 2
            samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), dims_feat(k)+1:dims_feat(k+1)) = Feats_enroll{i}.LBP{1}/norm(Feats_enroll{i}.LBP{1});
        elseif Feat_Type(k) == 3
            samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), dims_feat(k)+1:dims_feat(k+1)) = Feats_enroll{i}.HOG{1}/norm(Feats_enroll{i}.HOG{1});
        elseif Feat_Type(k) == 4
            samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), dims_feat(k)+1:dims_feat(k+1)) = Feats_enroll{i}.LPQ{1}/norm(Feats_enroll{i}.LPQ{1});
        end
        end
    sample_sub_id_genuine(s) = sample_sub_id_genuine(s) + 1;
end

%{
for i = 1:length(SubIDs)
    s_rank = find(SubIDs(i)==clientID);
    s = s_rank(1);
    if strcmp(PNLabels(i), 'P') % && strcmp(ALabels(i), 'IpN')  && strcmp(MLabels(i), 'FixN') && strcmp(FLabels(i), 'PhN')
        for k = 1:length(Feat_Type)
        if Feat_Type(k) == 1
            samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), dims_feat(k)+1:dims_feat(k+1)) = Feats{i}.MsLBP{1}/norm(Feats{i}.MsLBP{1});
        elseif Feat_Type(k) == 2
            samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), dims_feat(k)+1:dims_feat(k+1)) = Feats{i}.LBP{1}/norm(Feats{i}.LBP{1});
        elseif Feat_Type(k) == 3
            samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), dims_feat(k)+1:dims_feat(k+1)) = Feats{i}.HOG{1}/norm(Feats{i}.HOG{1});
        elseif Feat_Type(k) == 4
            samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), dims_feat(k)+1:dims_feat(k+1)) = Feats{i}.LPQ{1}/norm(Feats{i}.LPQ{1});
        end
        end
        samples_btype_subjects_genuine{s}(sample_sub_id_genuine(s)) = BLabels(i);       
        sample_sub_id_genuine(s) = sample_sub_id_genuine(s) + 1;        
        
    elseif strcmp(PNLabels(i), 'N') && sum(SubIDs(i) == clientID_source) > 0
        for k = 1:length(Feat_Type)
        if Feat_Type(k) == 1
            samples_data_subjects_fake{s}(sample_sub_id_fake(s), dims_feat(k)+1:dims_feat(k+1)) = Feats{i}.MsLBP{1}/norm(Feats{i}.MsLBP{1});
        elseif Feat_Type(k) == 2
            samples_data_subjects_fake{s}(sample_sub_id_fake(s), dims_feat(k)+1:dims_feat(k+1)) = Feats{i}.LBP{1}/norm(Feats{i}.LBP{1});
        elseif Feat_Type(k) == 3
            samples_data_subjects_fake{s}(sample_sub_id_fake(s), dims_feat(k)+1:dims_feat(k+1)) = Feats{i}.HOG{1}/norm(Feats{i}.HOG{1});
        elseif Feat_Type(k) == 4
            samples_data_subjects_fake{s}(sample_sub_id_fake(s), dims_feat(k)+1:dims_feat(k+1)) = Feats{i}.LPQ{1}/norm(Feats{i}.LPQ{1});
        end
        end
        samples_btype_subjects_fake{s}(sample_sub_id_fake(s)) = BLabels(i);
        samples_mtype_subjects_fake{s}(sample_sub_id_fake(s)) = MLabels(i);
        samples_atype_subjects_fake{s}(sample_sub_id_fake(s)) = ALabels(i);
        samples_ftype_subjects_fake{s}(sample_sub_id_fake(s)) = FLabels(i);
        sample_sub_id_fake(s) = sample_sub_id_fake(s) + 1;       
    end    
end
%}

% Assign samples for training
%{
BTypes_genuine = {'ContP' 'AdvP'};

BTypes_fake    = {'ContN' 'AdvN'};
MTypes_fake    = {'FixN' 'HandN'};
ATypes_fake    = {'IpN' 'MoN' 'PhN'};
FTypes_fake    = {'PhN' 'VidN'};

% Select 1/3 of the fake samples in the source subject domains
for i = 1:SUB_NUM_S  % we first assign for source subjects
    subid = clientID_source(i);
    s_rank = find(subid == clientID);
    s = s_rank(1);
    
    % reassign genuine samples: the first 1/3 
    samples_gen_data_sub = [];
    for b = 1:2
        ind = find(strcmp(samples_btype_subjects_genuine{s}, BTypes_genuine{b}));
        samples_gen_data_sub = [samples_gen_data_sub; samples_data_subjects_genuine{s}(ind(1:int16(length(ind)/3)), :)];
    end
    samples_data_subjects_genuine{s} = samples_gen_data_sub;
    
    % reassign fake samples
    samples_fake_data_sub = [];
    for b = 1:2
        for m = 1:2
            for a = 1:3
                for f = 1:2
                    ind = find(strcmp(samples_btype_subjects_fake{s}, BTypes_fake{b}) & strcmp(samples_mtype_subjects_fake{s}, MTypes_fake{m}) & ...
                                      strcmp(samples_atype_subjects_fake{s}, ATypes_fake{a}) & strcmp(samples_ftype_subjects_fake{s}, FTypes_fake{f}));
                    samples_fake_data_sub = [samples_fake_data_sub; samples_data_subjects_fake{s}(ind(1:int16(length(ind)/3)), :)];
                end
            end
        end
    end
    samples_data_subjects_fake{s} = samples_fake_data_sub;
end

for i = 1:SUB_NUM_T  % and then assign for target subjects
    subid = clientID_target(i);
    s_rank = find(subid == clientID);
    s = s_rank(1);
    
    % reassign genuine samples: the first 1/3 
    samples_gen_data_sub = [];
    for b = 1:2
        ind = find(strcmp(samples_btype_subjects_genuine{s}, BTypes_genuine{b}));
        samples_gen_data_sub = [samples_gen_data_sub; samples_data_subjects_genuine{s}(ind(1:int16(length(ind)/3)), :)];
    end
    samples_data_subjects_genuine{s} = samples_gen_data_sub;
end
%}

% Load virtual fake features of target subjects for training
QualityType = {'Enroll'};
if ~strcmp(method, 'NULL')
    VirtualFeatures = cell(length(QualityType), length(Feat_Type));
    for q = 1:length(QualityType)
        for k = 1:length(Feat_Type)
            VirtualFeatures{q, k} = load(strcat('..\Domain_Adaptation\SynthFeatures_AllTSubs_', QualityType{q}, '_', name{Feat_Type(k)}, '_', method));
        end
    end
end

% Assemble genuine (fake)  samples from all subjects
samples_data_train_fld_pos = [];
samples_data_train_fld_neg = [];

for s = 1:SUB_NUM
    samples_data_train_fld_pos = [samples_data_train_fld_pos; samples_data_subjects_genuine{s}(1:3:end, :)];
    samples_data_train_fld_neg = [samples_data_train_fld_neg; samples_data_subjects_fake{s}(1:3:end, :)];
end

if ~strcmp(method, 'NULL')
    for s = 1:SUB_NUM_T
        samples_data = [];
        for q = 1:size(VirtualFeatures, 1)
            samples_data_feat = [];
            for k = 1:size(VirtualFeatures, 2)
                
                data = VirtualFeatures{q, k}.SynthFeatures_AllTSubs{s};
                if isempty(data)
                    break;
                end
                data = bsxfun(@rdivide, data, sqrt(sum(data.^2, 2)));
                if k == 1
                    samples_data_feat = data;
                    continue;
                end
                if size(samples_data_feat, 1) > size(data, 1)
                    samples_data_feat = [samples_data_feat(1:size(data, 1), :), data];
                elseif size(samples_data_feat, 1) <= size(data, 1)
                    samples_data_feat = [samples_data_feat, data(1:size(samples_data_feat, 1), :)];
                end
               
            end
            samples_data = [samples_data; samples_data_feat];
        end
        samples_data_train_fld_neg = [samples_data_train_fld_neg; samples_data(1:3:end, :)];
    end
end

samples_data_fld_train  =  [samples_data_train_fld_pos; samples_data_train_fld_neg];
samples_label_fld_train =  [ones(size(samples_data_train_fld_pos, 1), 1); -1*ones(size(samples_data_train_fld_neg, 1), 1)];

% L2 Normalize feature vectors, has been normalized before
samples_data_fld_train = bsxfun(@rdivide, samples_data_fld_train, sqrt(sum(samples_data_fld_train.^2, 2)));


suffix = '';
for i = 1:length(QualityType)
    suffix = strcat(suffix, QualityType{i});
end

svm_param = ['-t 0 -c 10'];

% ---------------- train SVM model ---------------- %
model = svmtrain(samples_label_fld_train, samples_data_fld_train, svm_param);
       
% for s = 1:SUB_NUM
%     disp(strcat('Subject: ', num2str(s)));
%     
%     samples_data_train_fld_pos = samples_data_subjects_genuine{s};
%     samples_data_train_fld_neg = samples_data_subjects_fake{s};
% 
%     samples_data_fld_train  =  [samples_data_train_fld_pos; samples_data_train_fld_neg];
%     samples_label_fld_train =  [ones(size(samples_data_train_fld_pos, 1), 1); -1*ones(size(samples_data_train_fld_neg, 1), 1)];
% 
%     % L2 Normalize feature vectors, have been normalized before
%     % samples_data_fld_train = bsxfun(@rdivide, samples_data_fld_train, sqrt(sum(samples_data_fld_train.^2, 2)));
% 
%     % ------------- test trained SVM model by trainning data --------------- %
%     svmpredict(samples_label_fld_train,samples_data_fld_train,model);
% end

clear samples_data_fld_train;
clear samples_label_fld_train;

modelname = '';
for k = 1:length(Feat_Type)
    modelname = strcat(modelname, '_', name{Feat_Type(k)});
end

GenericModel = model;
save(strcat('GenericModel', modelname, '_', method, '_', suffix, '.mat'), 'GenericModel');

end

