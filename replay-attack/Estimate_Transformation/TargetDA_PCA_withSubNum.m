function TargetDA_PCA_withSubNum(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feats_enroll_SL, Labels_enroll_SL, Feat_Type, method)
%TARGETDA Summary of this function goes here
%   Function: Conduct target domain adaptation
%   Detailed explanation goes here
%   Input:
%        Feats_train_SL: feature set of first 20 subjects
%        Labels_train_SL: label set of first 20 subjects
%        Feats_test_SL: feature set of remaining 30 subjects
%        Labels_test_SL: feature set of remaning 30 subjects
%        QualityType_s: quality type of source subject for adaptation
%        QualityType_t: quality type of target subject for adaptation
%        method: the method for estimating transformation, the methods are
%        (1) Center-Shift, (2) OLS, (3) PLS

% Step 1: Organize fake training samples for the first 20 source subjects (first 1/3 part of all genuien samples)
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

samples_label_subjects = cell(1, SUB_NUM);
samples_label_subjects_gen = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    samples_label_subjects{s} = uint8(samples_label_subject);
end

% BLabel_gen = strrep(BLabel, 'N', 'P');

for i = 1:length(SubIDs)
    s_rank = find(SubIDs(i)==clientID);
    s = s_rank(1);
    if strcmp(MLabels(i), 'FixN') % && strcmp(ALabels(i), 'IpN') && strcmp(MLabels(i), 'FixN') && strcmp(FLabels(i), 'PhN')
        samples_label_subjects{s}(i) = 1;
    end
end

for i = 1:length(SubIDs_enroll)
    s_rank = find(SubIDs_enroll(i)==clientID_enroll);
    s = s_rank(1);
    samples_label_subjects_gen{s}(i) = 1;
end

samples_data_subjects = cell(1, SUB_NUM);         % save real and virtual fake samples for training
samples_data_subjects_gen = cell(1, SUB_NUM);     % save genuine samples for training

samples_btype_subjects = cell(1, SUB_NUM);
samples_atype_subjects = cell(1, SUB_NUM);
samples_ftype_subjects = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    samples_data_subjects{s} = zeros(sum(samples_label_subjects{s}), dims(Feat_Type));
    samples_data_subjects_gen{s} = zeros(sum(samples_label_subjects_gen{s}), dims(Feat_Type));
    
    samples_btype_subjects{s} = cell(sum(samples_label_subjects{s}), 1);
    samples_atype_subjects{s} = cell(sum(samples_label_subjects{s}), 1);
    samples_ftype_subjects{s} = cell(sum(samples_label_subjects{s}), 1);
end

sample_sub_id = ones(1, SUB_NUM);
sample_sub_id_gen = ones(1, SUB_NUM);

for i = 1:length(SubIDs)
    s_rank = find(SubIDs(i)==clientID);
    s = s_rank(1);
    if strcmp(MLabels(i), 'FixN') % && strcmp(ALabels(i), 'IpN')  && strcmp(MLabels(i), 'FixN') && strcmp(FLabels(i), 'PhN')
        if Feat_Type == 1
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats{i}.MsLBP{1};
        elseif Feat_Type == 2
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats{i}.LBP{1};
        elseif Feat_Type == 3
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats{i}.HOG{1};
        elseif Feat_Type == 4
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats{i}.LPQ{1};
        end
        samples_btype_subjects{s}(sample_sub_id(s)) = BLabels(i);
        samples_atype_subjects{s}(sample_sub_id(s)) = ALabels(i);
        samples_ftype_subjects{s}(sample_sub_id(s)) = FLabels(i);
        sample_sub_id(s) = sample_sub_id(s) + 1;
    end
end

for i = 1:length(SubIDs_enroll)
    s_rank = find(SubIDs_enroll(i)==clientID_enroll);
    s = s_rank(1);
    
    if Feat_Type == 1
        samples_data_subjects_gen{s}(sample_sub_id_gen(s), :) = Feats{i}.MsLBP{1};
    elseif Feat_Type == 2
        samples_data_subjects_gen{s}(sample_sub_id_gen(s), :) = Feats{i}.LBP{1};
    elseif Feat_Type == 3
        samples_data_subjects_gen{s}(sample_sub_id_gen(s), :) = Feats{i}.HOG{1};
    elseif Feat_Type == 4
        samples_data_subjects_gen{s}(sample_sub_id_gen(s), :) = Feats{i}.LPQ{1};
    end
    sample_sub_id_gen(s) = sample_sub_id_gen(s) + 1;
end

% Use only the fisrt 1/3 part of samples for estimating the transformation
BTypes = {'ContN' 'AdvN'};
ATypes = {'IpN' 'MoN' 'PhN'};
FTypes = {'PhN' 'VidN'};

for s = 1: SUB_NUM
    num_samples = size(samples_data_subjects_gen{s}, 1);
    samples_data_allsubjects = [];
    samples_data_allsubjects = [samples_data_allsubjects; samples_data_subjects_gen{s}(1:int16(num_samples / 4), :)];
    samples_data_allsubjects = [samples_data_allsubjects; samples_data_subjects_gen{s}(1 + int16(num_samples / 2):int16(3 * num_samples / 4), :)];    
    samples_data_subjects_gen{s} = bsxfun(@rdivide, samples_data_allsubjects, sqrt(sum(samples_data_allsubjects.^2, 2)));
end

% Compute the directional vectors from fake samples to genuine samples in
% one subject domain
global Transform;
if isempty(Transform)
    return;
end

SynthFeatures_AllTSubs = cell(SUB_NUM_T, SUB_NUM_S);

for SubNum = 1:SUB_NUM_S
    for a = 1:3
        samples_data_subject_matchings_fake = cell(1, SubNum);
        samples_data_allsubjects_matchings_fake = [];
        
        for i = 1:SubNum
            s = find(clientID_source(i) == clientID);
            s = s(1);
            samples_data_subject_genuine = samples_data_subjects_gen{s};
            
            ind = find(strcmp(samples_mtype_subjects{p}, MTypes{p}) & strcmp(samples_atype_subjects{p}, ATypes{a}) & strcmp(samples_ftype_subjects{p}, FTypes{f}));
            
            samples_data_subject_fake = samples_data_subjects{s}(ind(1:int16(length(ind)/3)), :); %samples_data_subjects{m};
            
            num_samples_genuine = size(samples_data_subject_genuine, 1);
            num_samples_fake = size(samples_data_subject_fake, 1);
            
            samples_data_subject_fake = [samples_data_subject_fake; repmat(mean(samples_data_subject_fake), [max(num_samples_genuine-num_samples_fake, 0) 1])];
            num_samples_fake = size(samples_data_subject_fake, 1);
            
            % find the matching fake samples for the genuine samples
            samples_data_subject_matchings_fake{i} = findMatchings(samples_data_subject_genuine, samples_data_subject_fake);
            if size(samples_data_subject_genuine, 1) > size(samples_data_subject_matchings_fake{i}, 1)
                samples_data_subject_matchings_fake{i} = [samples_data_subject_matchings_fake{i}; ...
                    samples_data_subject_fake(1:(size(samples_data_subject_genuine, 1) - size(samples_data_subject_matchings_fake{i}, 1)), :)];
            end
            samples_data_allsubjects_matchings_fake = [samples_data_allsubjects_matchings_fake; samples_data_subject_matchings_fake{i}];
            
        end
        
        
        samples_data_allsubject_matchings_fake_mean = mean(samples_data_allsubjects_matchings_fake, 1);
        
        samples_data_allsubjects_matchings_fake_centralized = bsxfun(@minus, samples_data_allsubjects_matchings_fake, samples_data_allsubject_matchings_fake_mean);
        
        
        % Step 2: load the linear weights for all target subjects
        
        % Step 3: Synthsize fake samples for target subjects
        
        for m = 1:SUB_NUM_T
            samples_synthdata_subject_train_fake = bsxfun(@plus, Transform{SubNum}.weights{m}'*samples_data_allsubjects_matchings_fake_centralized, ...
                samples_data_allsubject_matchings_fake_mean);
            SynthFeatures_AllTSubs{m, SubNum} = [SynthFeatures_AllTSubs{m, SubNum}; samples_synthdata_subject_train_fake];
        end
    end
end

save(strcat('SynthFeatures_AllTSubs_withSubNum_Enroll_', name{Feat_Type}, '_', method), 'SynthFeatures_AllTSubs', '-v7.3');

end

function samples_data_matched_fake = findMatchings(samples_data_subject_genuine, samples_data_subject_fake)

% Step 1-1: compute affinity matrix
AffinityMat = exp(-pdist2(samples_data_subject_genuine, samples_data_subject_fake));

% Step 1-2: SVD for affinity matrix
[U,S,V] = svd(AffinityMat);

% Step 1-3: find matchings for genuine samples of reference subjects
E = S;
E(logical(eye(min(size(S))))) = 1;
P = U*E*V';

% compute the mean value of similarities between two features in two groups
% derive the assignment
if size(P, 1) < size(P, 2)
    [V L] = max(P');
    samples_data_matched_fake = samples_data_subject_fake(int16(L), :);
else
    [V L] = max(P);
    samples_data_matched_fake = [samples_data_subject_fake; repmat(mean(samples_data_subject_fake, 1), size(P, 1)-size(P, 2))];
end

end

