function train_GenericModel_withSubNum(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Type)
%TRAIN_GENERICMODEL Summary of this function goes here
%   Detailed explanation goes here
%   Input:
%        Feats_train_SL: feature set of first 20 subjects
%        Labels_train_SL: label set of first 20 subjects
%        Feats_test_SL: feature set of remaining 30 subjects
%        Labels_test_SL: feature set of remaning 30 subjects
%        Feat_Type: type of feature used for face anti-spoofing

% Step 1: Organize genuine and fake training samples for the both source subjects and target subjects
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

% global samples_data_subjects_genuine;
% global samples_data_subjects_fake;

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
    samples_data_subjects_genuine{s} = zeros(sum(samples_label_subjects_genuine{s}), dims(Feat_Type));
    samples_data_subjects_fake{s} = zeros(sum(samples_label_subjects_fake{s}), dims(Feat_Type));    
    
    samples_stype_subjects_genuine{s} = cell(sum(samples_label_subjects_genuine{s}), 1);
    samples_stype_subjects_fake{s} = cell(sum(samples_label_subjects_fake{s}), 1);

    samples_qtype_subjects_genuine{s} = cell(sum(samples_label_subjects_genuine{s}), 1);
    samples_qtype_subjects_fake{s} = cell(sum(samples_label_subjects_fake{s}), 1);    
end

sample_sub_id_genuine = ones(1, SUB_NUM);
sample_sub_id_fake = ones(1, SUB_NUM);

for i = 1:length(SubIDs)
    s = SubIDs(i);
    if s <= 20
        if strcmp(TLabels(i), 'PP')
            if Feat_Type == 1
                samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), :) = Feats(i).MsLBP{1};
            elseif Feat_Type == 2
                samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), :) = Feats(i).LBP{1};
            elseif Feat_Type == 3
                samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), :) = Feats(i).HOG{1};
            elseif Feat_Type == 4
                samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), :) = Feats(i).LPQ{1};
            end
            samples_stype_subjects_genuine{s}(sample_sub_id_genuine(s)) = TLabels(i);
            samples_qtype_subjects_genuine{s}(sample_sub_id_genuine(s)) = QLabels(i);            
            sample_sub_id_genuine(s) = sample_sub_id_genuine(s) + 1;
        else
            if Feat_Type == 1
                samples_data_subjects_fake{s}(sample_sub_id_fake(s), :) = Feats(i).MsLBP{1};
            elseif Feat_Type == 2
                samples_data_subjects_fake{s}(sample_sub_id_fake(s), :) = Feats(i).LBP{1};                
            elseif Feat_Type == 3
                samples_data_subjects_fake{s}(sample_sub_id_fake(s), :) = Feats(i).HOG{1};
            elseif Feat_Type == 4
                samples_data_subjects_fake{s}(sample_sub_id_fake(s), :) = Feats(i).LPQ{1};
            end
            samples_stype_subjects_fake{s}(sample_sub_id_fake(s)) = TLabels(i);
            samples_qtype_subjects_fake{s}(sample_sub_id_fake(s)) = QLabels(i);    
            sample_sub_id_fake(s) = sample_sub_id_fake(s) + 1;
        end
    elseif s > 20 && s <= SUB_NUM
        if strcmp(TLabels(i), 'PP')
            if Feat_Type == 1
                samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), :) = Feats(i).MsLBP{1};
            elseif Feat_Type == 2
                samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), :) = Feats(i).LBP{1};
            elseif Feat_Type == 3
                samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), :) = Feats(i).HOG{1};
            elseif Feat_Type == 4
                samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), :) = Feats(i).LPQ{1};
            end
            samples_stype_subjects_genuine{s}(sample_sub_id_genuine(s)) = TLabels(i);
            samples_qtype_subjects_genuine{s}(sample_sub_id_genuine(s)) = QLabels(i);   
            sample_sub_id_genuine(s) = sample_sub_id_genuine(s) + 1;
        end
    end
end

% Use only the fisrt 1/3 part of samples for training person-specific face anti-spoofing classifier
QualityTypes_Genuine = {'LP' 'MP' 'HP'};
QualityTypes_Fake = {'HN' 'MN' 'LN'};
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


% Assemble genuine (fake)  samples from all subjects
GenericModel = cell(SUB_NUM_S, 1);

for SubNum = 1:SUB_NUM_S
    samples_data_train_fld_pos = [];
    samples_data_train_fld_neg = [];
    
    for s = 1:SubNum
        samples_data_train_fld_pos = [samples_data_train_fld_pos; samples_data_subjects_genuine{s}];
        samples_data_train_fld_neg = [samples_data_train_fld_neg; samples_data_subjects_fake{s}];
    end
    
    for s = SUB_NUM_S + 1:SUB_NUM
        samples_data_train_fld_pos = [samples_data_train_fld_pos; samples_data_subjects_genuine{s}];
    end
    
    samples_data_fld_train  =  [samples_data_train_fld_pos; samples_data_train_fld_neg];
    samples_label_fld_train =  [ones(size(samples_data_train_fld_pos, 1), 1); -1*ones(size(samples_data_train_fld_neg, 1), 1)];
    
    % L2 Normalize feature vectors
    samples_data_fld_train = bsxfun(@rdivide, samples_data_fld_train, sqrt(sum(samples_data_fld_train.^2, 2)));
    
    % svm_param = ['-t 0 -c 0.5'];
    t = 0;
    c = 100;
    
    svm_param = strcat('-t ',num2str(t), ' -c ', num2str(c), ' -g 0.5 -h 0');
    
    % ---------------- train SVM model ---------------- %
    model = svmtrain(samples_label_fld_train, samples_data_fld_train, svm_param);
    
    GenericModel{SubNum} = model;
    
    clear samples_data_fld_train;
    clear samples_label_fld_train;
    
end

if ~exist(strcat('GenericModel_withSubNum_', name{Feat_Type}, '.mat'))
    save(strcat('GenericModel_withSubNum_', name{Feat_Type}, '.mat'), 'GenericModel');
end

end

