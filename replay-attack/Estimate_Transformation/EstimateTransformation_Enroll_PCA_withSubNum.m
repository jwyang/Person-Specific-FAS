function EstimateTransformation_Enroll_PCA_withSubNum(Feats_enroll_SL, Labels_train_SL, Labels_devel_SL, Labels_test_SL, Labels_enroll_SL, Feat_Type, BLabel, method)
%ESTIMATETRANSFORMATION_ITERATIVE Summary of this function goes here
%   Function: Estimate transformation from one subject domain to another one iteratively
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


% Step 1: Organize genuine training samples for all 20+30 subjects (first 1/3 part of all genuien samples)
name = {'MsLBP' 'LBP' 'HOG' 'LPQ'};
dims = [833 361 378 256];

SubIDs_train  = Labels_train_SL.SubID_train;
%{
PNLabels_train = Labels_train_SL.PNLabels_train;
BLabels_train = Labels_train_SL.BLabels_train;
MLabels_train = Labels_train_SL.MLabels_train;
ALabels_train = Labels_train_SL.ALabels_train;
FLabels_train = Labels_train_SL.FLabels_train;
Feats_train   = Feats_train_SL.Feats_train;
%}

SubIDs_devel   = Labels_devel_SL.SubID_devel;
%{
PNLabels_devel  = Labels_devel_SL.PNLabels_devel;
BLabels_devel  = Labels_devel_SL.BLabels_devel;
MLabels_devel  = Labels_devel_SL.MLabels_devel;
ALabels_devel = Labels_devel_SL.ALabels_devel;
FLabels_devel = Labels_devel_SL.FLabels_devel;
Feats_devel    = Feats_devel_SL.Feats_devel;
%}

SubIDs_test   = Labels_test_SL.SubID_test;
%{
PNLabels_test  = Labels_test_SL.PNLabels_test;
BLabels_test  = Labels_test_SL.BLabels_test;
MLabels_test  = Labels_test_SL.MLabels_test;
ALabels_test = Labels_test_SL.ALabels_test;
FLabels_test = Labels_test_SL.FLabels_test;
Feats_test    = Feats_test_SL.Feats_test;
%}

SubIDs_enroll   = Labels_enroll_SL.SubID_enroll;
PNLabels_enroll  = Labels_enroll_SL.PNLabels_enroll;
BLabels_enroll  = Labels_enroll_SL.BLabels_enroll;
MLabels_enroll  = Labels_enroll_SL.MLabels_enroll;
ALabels_enroll = Labels_enroll_SL.ALabels_enroll;
FLabels_enroll = Labels_enroll_SL.FLabels_enroll;
Feats_enroll    = Feats_enroll_SL.Feats_enroll;

% concatenate train and enroll information
SubIDs = SubIDs_enroll;
PNLabels = PNLabels_enroll;
BLabels = BLabels_enroll;
MLabels = MLabels_enroll;
ALabels = ALabels_enroll;
FLabels = FLabels_enroll;
Feats   = Feats_enroll;


clientID = unique(SubIDs);

clientID_source  = unique(SubIDs_train);
clientID_target = unique([SubIDs_devel, SubIDs_test]);

SUB_NUM_S = length(clientID_source);
SUB_NUM_T = length(clientID_target);
SUB_NUM = SUB_NUM_S + SUB_NUM_T;

samples_label_subject = zeros(length(SubIDs), 1);
samples_label_subjects = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    samples_label_subjects{s} = uint8(samples_label_subject);
end

for i = 1:length(SubIDs)
    s_rank = find(SubIDs(i)==clientID);
    s = s_rank(1);
    if strcmp(BLabels(i), BLabel) % && strcmp(ALabels(i), 'IpN') && strcmp(MLabels(i), 'FixN') && strcmp(FLabels(i), 'PhN')
        samples_label_subjects{s}(i) = 1;
    end
end

samples_data_subjects = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    samples_data_subjects{s} = zeros(sum(samples_label_subjects{s}), dims(Feat_Type));
end

sample_sub_id = ones(1, SUB_NUM);

for i = 1:length(SubIDs)
    s_rank = find(SubIDs(i)==clientID);
    s = s_rank(1);
    if strcmp(BLabels(i), BLabel) % && strcmp(ALabels(i), 'IpN')  && strcmp(MLabels(i), 'FixN') && strcmp(FLabels(i), 'PhN')
        if Feat_Type == 1
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats{i}.MsLBP{1};
        elseif Feat_Type == 2
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats{i}.LBP{1};
        elseif Feat_Type == 3
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats{i}.HOG{1};
        elseif Feat_Type == 4
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats{i}.LPQ{1};
        end
        sample_sub_id(s) = sample_sub_id(s) + 1;
    end
end

% Use only the first 1/3 part of samples for estimating the transformation
%{
for s = 1:SUB_NUM
    samples_data_subjects{s} = samples_data_subjects{s}(1:int16(sample_sub_id(s)/3), :);
end
%}

% Step 2: Esitmate the PCA coefficients (Xiaoou Tang's paper)

% Step 1: Obtain the feature vectors for all training subjects
Transform = cell(SUB_NUM_S, 1);

for SubNum = 1:SUB_NUM_S
    
    samples_data_allsubjects = [];
    
    for i = 1:SubNum
        s = find(clientID_source(i) == clientID);
        s = s(1);
        num_samples = size(samples_data_subjects{s}, 1);
        samples_data_allsubjects = [samples_data_allsubjects; samples_data_subjects{s}(1:int16(num_samples / 4), :)];
        samples_data_allsubjects = [samples_data_allsubjects; samples_data_subjects{s}(1 + int16(num_samples / 2):int16(3 * num_samples / 4), :)];
    end
    
    % Step 2: Centralize training samples
    samples_data_allsubjects_mean = mean(samples_data_allsubjects, 1);
    samples_data_allsubjects_centralized = bsxfun(@minus, samples_data_allsubjects, samples_data_allsubjects_mean);
    
    
    % Step 3: Compute Eigenface, Eigenvalue and Eigenfaces
    
    X = samples_data_allsubjects_centralized*samples_data_allsubjects_centralized';
    [eigvectors, eigvalues] = eig(X);
    
    % Step 3-1: determine the dimension of sub-space K
    eigvalues_vector = diag(eigvalues)';
    eigvalues_vector_cumsum = cumsum(abs(eigvalues_vector))/sum(abs(eigvalues_vector));
    ind = find(eigvalues_vector_cumsum < 0.02);
    K = ind(end);
    
    E = samples_data_allsubjects_centralized'*eigvectors(:, K:end)*sqrt(inv(eigvalues(K:end, K:end)));
    
    % Ste 4: Compute linear weights
    
    
    % Step 4-2: Compute the weight factors
    
    c = eigvectors(:, K:end)*sqrt(inv(eigvalues(K:end, K:end)))*E';
    
    
    % Compute the lienar weights for target sujbects (have genuine faces but not fake faces)
    
    samples_weight_subjects = cell(1, SUB_NUM_T);
    
    for i = 1:SUB_NUM_T
        s = find(clientID_target(i) == clientID);
        s = s(1);
        samples_data_subject = samples_data_subjects{s};
        % Step 1: centralize the samples for target subject
        samples_data_subject_centralize = bsxfun(@minus, samples_data_subject, samples_data_allsubjects_mean);
        % Step 2: computer linear weights for all samples of target subjects
        samples_weight_subjects{i} = c*samples_data_subject_centralize';
    end
    
    Transform{SubNum}.weights = samples_weight_subjects;
    Transform{SubNum}.mean  = samples_data_allsubjects_mean;
end

save(strcat('Transform_withSubNum_', BLabel, '_', name{Feat_Type}, '_', method, '.mat'), 'Transform', '-v7.3');

end