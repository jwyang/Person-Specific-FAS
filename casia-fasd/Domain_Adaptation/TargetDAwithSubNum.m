function TargetDAwithSubNum(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Type, QualityType, method)
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

QualityType_gen = strrep(QualityType, 'N', 'P');

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
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats(i).MsLBP{1};
        elseif Feat_Type == 2
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats(i).LBP{1};
        elseif Feat_Type == 3
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats(i).HOG{1};
        elseif Feat_Type == 4
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats(i).LPQ{1};
        end
        samples_type_subjects{s}(sample_sub_id(s)) = TLabels(i);
        sample_sub_id(s) = sample_sub_id(s) + 1;
    elseif strcmp(QLabels(i), QualityType_gen)
        if Feat_Type == 1
            samples_data_subjects_gen{s}(sample_sub_id_gen(s), :) = Feats(i).MsLBP{1};
        elseif Feat_Type == 2
            samples_data_subjects_gen{s}(sample_sub_id_gen(s), :) = Feats(i).LBP{1};
        elseif Feat_Type == 3
            samples_data_subjects_gen{s}(sample_sub_id_gen(s), :) = Feats(i).HOG{1};
        elseif Feat_Type == 4
            samples_data_subjects_gen{s}(sample_sub_id_gen(s), :) = Feats(i).LPQ{1};
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
    % L2-Normalize the features in source subject domain
    samples_train = bsxfun(@rdivide, samples_train, sqrt(sum(samples_train.^2, 2)));    
    samples_data_subjects{s} = samples_train; % samples_data_subjects{s}(1:int16(sample_sub_id(s)/3), :);
end

for s = 1: SUB_NUM
    num = size(samples_data_subjects_gen{s}, 1);
    samples_data_subjects_gen{s} = samples_data_subjects_gen{s}(1:int16(num/3), :);
    samples_data_subjects_gen{s} = bsxfun(@rdivide, samples_data_subjects_gen{s}, sqrt(sum(samples_data_subjects_gen{s}.^2, 2)));
end

% Compute the directional vectors from fake samples to genuine samples in
% one subject domain
directVectors = zeros(SUB_NUM_S, dims(Feat_Type));
for s = 1:SUB_NUM_S
    directVectors(s, :) = mean(samples_data_subjects_gen{s}, 1) - mean(samples_data_subjects{s}, 1);
    % directVectors(s, :) = directVectors(s, :)./norm(directVectors(s, :));
end

% Step 2: Synthesize features of unobserved fake samples for target subjects
% Step 2-1: Load transformations
global Transform;
global Transform_CS;

if isempty(Transform)
    % if strcmp(QualityType, 'MP')
    %    load(strcat('..\Estimate_Transfomration_for_Expe\Transform_MP_MP_', name{Feat_Type}, '_', method, '.mat'));    
    % else
        load(strcat('..\Estimate_Transfomration_for_Expe\Transform_', QualityType, '_', QualityType, '_', name{Feat_Type}, '_', method, '.mat'));
    % end
end

SynthFeatures_AllTSubs = cell(SUB_NUM_T, SUB_NUM_S);

tic;
for SubNum = 1:SUB_NUM_S
    parfor n_s = 1:SUB_NUM_T
        n = n_s + SUB_NUM_S;
        SynthFeatures_TSub = [];
        for m = 1:SubNum
            
            samples_source = samples_data_subjects{m};
            
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
            if min(distmat(:)) < dist_thresholds(Feat_Type) && strcmp(QualityType, 'HN')
                continue;
            else
                SynthFeatures_TSub = [SynthFeatures_TSub; synthdata'];
            end
        end
        
        if ~strcmp(QualityType, 'HN')
            SynthFeatures_AllTSubs{n_s, SubNum} = SynthFeatures_TSub(1:SubNum:end, :);
        else
            SynthFeatures_AllTSubs{n_s, SubNum} = SynthFeatures_TSub(1:end, :);
        end
    end
end
toc;
save(strcat('SynthFeatures_AllTSubs_withSubNum_', QualityType, '_', name{Feat_Type}, '_', method), 'SynthFeatures_AllTSubs', '-v7.3');

end

