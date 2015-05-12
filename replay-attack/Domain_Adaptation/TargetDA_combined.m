function TargetDA_combined(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feats_enroll_SL, Labels_enroll_SL, Feat_Type, method)
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

samples_mtype_subjects = cell(1, SUB_NUM);
samples_atype_subjects = cell(1, SUB_NUM);
samples_ftype_subjects = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    samples_data_subjects{s} = zeros(sum(samples_label_subjects{s}), dims(Feat_Type));
    samples_data_subjects_gen{s} = zeros(sum(samples_label_subjects_gen{s}), dims(Feat_Type));
    
    samples_mtype_subjects{s} = cell(sum(samples_label_subjects{s}), 1);
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
        samples_mtype_subjects{s}(sample_sub_id(s)) = MLabels(i);
        samples_atype_subjects{s}(sample_sub_id(s)) = ALabels(i);
        samples_ftype_subjects{s}(sample_sub_id(s)) = FLabels(i);
        sample_sub_id(s) = sample_sub_id(s) + 1;
    end
end


for i = 1:length(SubIDs_enroll)
    s_rank = find(SubIDs_enroll(i)==clientID_enroll);
    s = s_rank(1);
    
    if Feat_Type == 1
        samples_data_subjects_gen{s}(sample_sub_id_gen(s), :) = Feats_enroll{i}.MsLBP{1};
    elseif Feat_Type == 2
        samples_data_subjects_gen{s}(sample_sub_id_gen(s), :) = Feats_enroll{i}.LBP{1};
    elseif Feat_Type == 3
        samples_data_subjects_gen{s}(sample_sub_id_gen(s), :) = Feats_enroll{i}.HOG{1};
    elseif Feat_Type == 4
        samples_data_subjects_gen{s}(sample_sub_id_gen(s), :) = Feats_enroll{i}.LPQ{1};
    end
    sample_sub_id_gen(s) = sample_sub_id_gen(s) + 1;
    
end

clear Feats;
clear Feats_enroll;
% Use only the fisrt 1/3 part of samples for estimating the transformation
% There are overall 12 spoofing types for each subject under a background configuration
% Select 1/3 of the fake samples in the source subject domains
% for i = 1:SUB_NUM_S
%     subid = clientID_source(i);
%     s_rank = find(subid == clientID);
%     s = s_rank(1);
%
%     samples_train = [];
%     % Add the first 1/3 part of samples under a spoofing type into training
%     for m = 1:2
%         for a = 1:3
%             for f = 1:2
%                 ind = find(strcmp(samples_mtype_subjects{s}, MTypes{m}) & strcmp(samples_atype_subjects{s}, ATypes{a}) & strcmp(samples_ftype_subjects{s}, FTypes{f}));
%                 samples_train = [samples_train; samples_data_subjects{s}(ind(1:int16(length(ind)/3)), :)];
%             end
%         end
%     end
%
%     % L2-Normalize the features in source subject domain
%     samples_train = bsxfun(@rdivide, samples_train, sqrt(sum(samples_train.^2, 2)));
%     samples_data_subjects{s} = samples_train; % samples_data_subjects{s}(1:int16(sample_sub_id(s)/3), :);
% end

for s = 1:SUB_NUM
    % num = size(samples_data_subjects_gen{s}, 1);
    % samples_data_subjects_gen{s} = samples_data_subjects_gen{s}(1:int16(num/3), :);
    samples_data_subjects_gen{s} = bsxfun(@rdivide, samples_data_subjects_gen{s}, sqrt(sum(samples_data_subjects_gen{s}.^2, 2)));
end

% Step 2: Synthesize features of unobserved fake samples for target subjects
% one subject domain
dist_thresholds = [0.2 0.2 0.4 0.2];

global Transform;
global Transform_CS;

SynthFeatures_AllTSubs = cell(SUB_NUM_T, 1);

for n_s = 1:SUB_NUM_T
    n = find(clientID_target(n_s) == clientID);
    n = n(1);
    
    SynthFeatures_TSub = [];
    for m_s = 1:SUB_NUM_S
        m = find(clientID_source(m_s) == clientID);
        m = m(1);
        samples_source = samples_data_subjects{m}; %samples_data_subjects{m};
        % L2-Normalize the features in source subject domain
        samples_source = bsxfun(@rdivide, samples_source, sqrt(sum(samples_source.^2, 2)));
        
        if ~strcmp(method, 'CS')  % if use OLS or PLS for adaptation, then translate the center
            H_mn = Transform.H{m, n};
            T_mn = Transform.T{m, n};
            samples_source_centralized = bsxfun(@minus, samples_source, mean(samples_source, 1));
            synthdata = bsxfun(@plus, H_mn*samples_source_centralized', T_mn(:, 1));
            synthdata = bsxfun(@plus, synthdata, mean(samples_source', 2) - mean(synthdata, 2));
            
            H_mn = Transform_CS.H{m, n};
            T_mn = Transform_CS.T{m, n};
            synthdata = bsxfun(@plus, H_mn*synthdata, T_mn(:, 1));
        else
            H_mn = Transform.H{m, n};
            T_mn = Transform.T{m, n};
            synthdata = bsxfun(@plus, H_mn*samples_source', T_mn(:, 1));
        end
        % synthdata(synthdata(:) < 0) = 0;
        % Compute the sitance between the synthesized samples and the genuine
        % samples from the same subject. If it is too close, then we should not
        % add the virtual features into the training set.
        synthdata = bsxfun(@rdivide, synthdata, sqrt(sum(synthdata.^2, 1)));
        distmat = pdist2(samples_data_subjects_gen{n}, synthdata');
        
        % direct_vec = (mean(samples_data_subjects_gen{n}, 1) - mean(synthdata', 1));
        %  direct_vec = direct_vec./norm(direct_vec);
        % if the direction of virtual features violate the directions in the
        % source domains, then remove it
        distmat_min = min(distmat);
        ind = distmat_min >= dist_thresholds(Feat_Type);
        
        % if min(distmat(:)) < dist_thresholds(Feat_Type)
        %     continue;
        % else
            SynthFeatures_TSub = [SynthFeatures_TSub; synthdata(:, ind)'];
        % end
    end
    SynthFeatures_AllTSubs{n_s} = SynthFeatures_TSub(1:SUB_NUM_S:end, :);
end

save(strcat('SynthFeatures_AllTSubs_Enroll_', name{Feat_Type}, '_', method), 'SynthFeatures_AllTSubs', '-v7.3');

end

