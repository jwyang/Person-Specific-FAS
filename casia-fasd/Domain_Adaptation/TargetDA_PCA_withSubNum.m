function TargetDA_PCA_withSubNum(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Type, QualityType, method)
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
dims = [833 361 378 256];
dist_thresholds = [0.3 0.2 0.4 0.2];

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

samples_label_subject = zeros(length(SubIDs), 1);
samples_label_subjects = cell(1, SUB_NUM);

% for the genuine samples of target subjects in the training set.

samples_label_subjects_gen = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    samples_label_subjects{s} = uint8(samples_label_subject);
    samples_label_subjects_gen{s} = uint8(samples_label_subject);
end

QualityType_gen = 'HP'; % strrep(QualityType, 'N', 'P');

for i = 1:length(SubIDs)
    s = SubIDs(i);
    if s <= SUB_NUM_S && strcmp(QLabels(i), QualityType)
        samples_label_subjects{s}(i) = 1;
    elseif strcmp(QLabels(i), QualityType_gen)
        samples_label_subjects_gen{s}(i) = 1;
    end
end

samples_data_subjects = cell(1, SUB_NUM);

samples_data_subjects_gen = cell(1, SUB_NUM);

samples_type_subjects = cell(1, SUB_NUM);

samples_type_subjects_gen = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    samples_data_subjects{s} = zeros(sum(samples_label_subjects{s}), dims(Feat_Type));
    samples_data_subjects_gen{s} = zeros(sum(samples_label_subjects_gen{s}), dims(Feat_Type));
    
    samples_type_subjects{s} = cell(sum(samples_label_subjects{s}), 1);
    samples_type_subjects_gen{s} = cell(sum(samples_label_subjects{s}), 1);
end

sample_sub_id = ones(1, SUB_NUM);
sample_sub_id_gen = ones(1, SUB_NUM);

for i = 1:length(SubIDs)
    s = SubIDs(i);
    if s <= SUB_NUM_S && strcmp(QLabels(i), QualityType)  % Select ths samples in source domain whose quality type is QualityType_s
        if Feat_Type == 1
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats(i).MsLBP{1}/norm(Feats(i).MsLBP{1});
        elseif Feat_Type == 2
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats(i).LBP{1}/norm(Feats(i).LBP{1});
        elseif Feat_Type == 3
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats(i).HOG{1}/norm(Feats(i).HOG{1});
        elseif Feat_Type == 4
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats(i).LPQ{1}/norm(Feats(i).LPQ{1});
        end
        samples_type_subjects{s}(sample_sub_id(s)) = TLabels(i);
        sample_sub_id(s) = sample_sub_id(s) + 1;
    elseif strcmp(QLabels(i), QualityType_gen)
        if Feat_Type == 1
            samples_data_subjects_gen{s}(sample_sub_id_gen(s), :) = Feats(i).MsLBP{1}/norm(Feats(i).MsLBP{1});
        elseif Feat_Type == 2
            samples_data_subjects_gen{s}(sample_sub_id_gen(s), :) = Feats(i).LBP{1}/norm(Feats(i).LBP{1});
        elseif Feat_Type == 3
            samples_data_subjects_gen{s}(sample_sub_id_gen(s), :) = Feats(i).HOG{1}/norm(Feats(i).HOG{1});
        elseif Feat_Type == 4
            samples_data_subjects_gen{s}(sample_sub_id_gen(s), :) = Feats(i).LPQ{1}/norm(Feats(i).LPQ{1});
        end
        
        sample_sub_id_gen(s) = sample_sub_id_gen(s) + 1;
    end
end

% Use only the fisrt 1/3 part of samples for estimating the transformation
STypes = {'CPN' 'WPN' 'VN'};
for s = 1:SUB_NUM_S
    samples_train = [];
    % Add the first 1/3 part of samples under a spoofing type into training
    for t = 1:3
        ind = find(strcmp(samples_type_subjects{s}, STypes{t}));
        samples_train = [samples_train; samples_data_subjects{s}(ind(1:int16(length(ind)/3)), :)];
    end
    % L2-Normalize the features in source subject domain, has normalized before
    % samples_train = bsxfun(@rdivide, samples_train, sqrt(sum(samples_train.^2, 2)));
    samples_data_subjects{s} = samples_train; % samples_data_subjects{s}(1:int16(sample_sub_id(s)/3), :);
end

for s = 1: SUB_NUM
    num = size(samples_data_subjects_gen{s}, 1) + 1;
    samples_data_subjects_gen{s} = samples_data_subjects_gen{s}(1:int16(num/3), :);
    samples_data_subjects_gen{s} = bsxfun(@rdivide, samples_data_subjects_gen{s}, sqrt(sum(samples_data_subjects_gen{s}.^2, 2)));
end

% Compute the directional vectors from fake samples to genuine samples in
% one subject domain
samples_data_subject_matchings_fake = cell(1, SUB_NUM_S);

ratios = zeros(SUB_NUM_S, 1);

SynthFeatures_AllTSubs = cell(SUB_NUM_T, SUB_NUM_S);
global Transform;

for SubNum = 1:SUB_NUM_S
    samples_patches_allsubs = cell(SubNum, 1);
    for s = 1:SubNum
        samples_data_subject_genuine = samples_data_subjects_gen{s};
        samples_data_subject_fake = samples_data_subjects{s};
        
        num_samples_genuine = size(samples_data_subject_genuine, 1);
        num_samples_fake = size(samples_data_subject_fake, 1);
        
        samples_data_subject_fake = [samples_data_subject_fake; repmat(mean(samples_data_subject_fake), [max(num_samples_genuine-num_samples_fake, 0) 1])];
        num_samples_fake = size(samples_data_subject_fake, 1);
        
        % find the matching fake samples for the genuine samples
        ratio = floor(num_samples_fake/num_samples_genuine);
        if ratio ~= 0
            samples_patches_allsubs{s} = cell(ratio, 1);
            ratios(s) = ratio;
            for r = 1:ratio
                % samples_patch = samples_data_subject_fake((r-1)*num_samples_genuine+1:r*num_samples_genuine, :);
                samples_patches_allsubs{s}{r} = findMatchings(samples_data_subject_genuine, samples_data_subject_fake);
            end
        else
            ratio = 1;
            samples_patches_allsubs{s} = cell(ratio, 1);
            samples_patches_allsubs{s}{1} = [samples_data_subject_fake; ...
                samples_data_subject_fake(1:(size(samples_data_subject_genuine, 1) - size(samples_data_subject_matchings_fake{s}, 1)), :)];
        end
    end
    
    
    samples_data_allsubjects_matchings_fake = [];
    
    for i = 1:SubNum
        samples_data_subject_matchings_fake{i} = samples_patches_allsubs{i}{1};        
        samples_data_allsubjects_matchings_fake = [samples_data_allsubjects_matchings_fake; samples_data_subject_matchings_fake{i}];
    end
    
    samples_data_allsubject_matchings_fake_mean = mean(samples_data_allsubjects_matchings_fake, 1);
    
    samples_data_allsubjects_matchings_fake_centralized = bsxfun(@minus, samples_data_allsubjects_matchings_fake, samples_data_allsubject_matchings_fake_mean);
    
    
    % Step 2: load the linear weights for all target subjects
    
    % Step 3: Synthsize fake samples for target subjects
    
    for m = SUB_NUM_S+1:SUB_NUM
        samples_synthdata_subject_train_fake = bsxfun(@plus, Transform{SubNum}.weights{m-SUB_NUM_S}'*samples_data_allsubjects_matchings_fake_centralized, ...
            samples_data_allsubject_matchings_fake_mean);
        SynthFeatures_AllTSubs{m-SUB_NUM_S, SubNum} = [SynthFeatures_AllTSubs{m-SUB_NUM_S, SubNum}; samples_synthdata_subject_train_fake];
    end
end

save(strcat('SynthFeatures_AllTSubs_withSubNum_', QualityType, '_', name{Feat_Type}, '_', method), 'SynthFeatures_AllTSubs', '-v7.3');

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

