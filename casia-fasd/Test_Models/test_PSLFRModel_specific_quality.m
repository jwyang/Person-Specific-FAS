function test_PSLFRModel_specific_quality(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Type, Quality_Type_train, Quality_Type_test, method)
%TEST_PSMODELS Summary of this function goes here
%   Detailed explanation goes here
%   Input:
%        Feats_train_SL: feature set of first 20 subjects
%        Labels_train_SL: label set of first 20 subjects
%        Feats_test_SL: feature set of remaining 30 subjects
%        Labels_test_SL: feature set of remaning 30 subjects
%        Feat_Type: type of feature used for face anti-spoofing
%        method: the method for estimating transformation, the methods are
%        (1) Center-Shift, (2) OLS, (3) PLS

% Step 1: Organize genuine and fake training samples for the both source subjects and target subjects
dims = [833 361 378 256];
name = {'MsLBP' 'LBP' 'HOG' 'LPQ'};
QualityType_Gen = strcat(Quality_Type_test, 'P');
QualityType_Fake = strcat(Quality_Type_test, 'N');

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
    if strcmp(QLabels(i), QualityType_Gen)
        samples_label_subjects_genuine{s}(i) = 1;
    elseif strcmp(QLabels(i), QualityType_Fake)
        samples_label_subjects_fake{s}(i) = 1;
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
    if sum(strcmp(QLabels(i), QualityType_Gen)) > 0
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
    elseif sum(strcmp(QLabels(i), QualityType_Fake)) > 0
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
end

% Use only the fisrt 1/3 part of samples for training person-specific face anti-spoofing classifier
%{
QualityTypes_Genuine = {'LP' 'MP' 'HP'};
QualityTypes_Fake = {'HN' 'MN' 'LN'};
SpoofingTypes = {'CPN' 'WPN' 'VN'};

for s = 1:SUB_NUM
    % reassign genuine samples
    samples_gen_data_sub = [];
    for q = 1:3
        ind = find(strcmp(samples_qtype_subjects_genuine{s}, QualityTypes_Genuine{q}));
        samples_gen_data_sub = [samples_gen_data_sub; samples_data_subjects_genuine{s}(ind(int16(2*length(ind)/3)+1:end), :)];
    end
    samples_data_subjects_genuine{s} = samples_gen_data_sub;
    
    if s <= SUB_NUM_S
        % reassign fake samples for development in source subject domains
        samples_fake_data_sub = [];
        for a = 1:3
            for q = 1:3
                ind = find(strcmp(samples_stype_subjects_fake{s}, SpoofingTypes{a}) & strcmp(samples_qtype_subjects_fake{s}, QualityTypes_Fake{q}));
                samples_fake_data_sub = [samples_fake_data_sub; samples_data_subjects_fake{s}(ind(1+int16(2*length(ind)/3):end), :)];
            end
        end
        samples_data_subjects_fake{s} = samples_fake_data_sub;
    else
        % reassign fake samples for development in target subject domains
        samples_fake_data_sub = [];
        for a = 1:3
            for q = 1:3
                ind = find(strcmp(samples_stype_subjects_fake{s}, SpoofingTypes{a}) & strcmp(samples_qtype_subjects_fake{s}, QualityTypes_Fake{q}));
                samples_fake_data_sub = [samples_fake_data_sub; samples_data_subjects_fake{s}(ind(1+int16(length(ind)/2):end), :)];
            end
        end
        samples_data_subjects_fake{s} = samples_fake_data_sub;
    end
end
%}

% Use only the fisrt 1/2 part of samples for training person-specific face anti-spoofing classifier
SpoofingTypes = {'CPN' 'WPN' 'VN'};

for s = 1:SUB_NUM
    % reassign genuine samples
    % samples_gen_data_sub = [];
    % for q = 1:length(QualityType_Gen)
    %    ind = find(strcmp(samples_qtype_subjects_genuine{s}, QualityType_Gen));
    %    samples_gen_data_sub = [samples_gen_data_sub; samples_data_subjects_genuine{s}(1:int16(length(ind)/2), :)];  % the first half samples
    % end
    ind = find(strcmp(samples_qtype_subjects_genuine{s}, QualityType_Gen));
    samples_data_subjects_genuine{s} = samples_data_subjects_genuine{s}(1+int16(length(ind)/2):end, :);  % the first half samples
    
    
    % reassign fake samples for development in target subject domains
    samples_fake_data_sub = [];
    % for q = 1:length(QualityType_Fake)
        for a = 1:length(SpoofingTypes)        
            ind = find(strcmp(samples_stype_subjects_fake{s}, SpoofingTypes{a}) & strcmp(samples_qtype_subjects_fake{s}, QualityType_Fake));
            samples_fake_data_sub = [samples_fake_data_sub; samples_data_subjects_fake{s}(ind(1+int16(length(ind)/2):end), :)];
        end
    % end
    samples_data_subjects_fake{s} = samples_fake_data_sub;
    
end

suffix = '';
for i = 1:length(Quality_Type_train)
    suffix = strcat(suffix, Quality_Type_train{i});
end

modelname = '';
for k = 1:length(Feat_Type)
    modelname = strcat(modelname, '_', name{Feat_Type(k)});
end

% Load anti-spoofing models
load(strcat('..\Train_Models\PSModels', modelname, '_', method, '_', suffix, '.mat'));

% Load thrsholds for computing HTER
load(strcat('..\Develop_Models\PerSpecPerf_devel', modelname, '_', method, '_', suffix, '_', Quality_Type_test, '.mat'));

% Compute HTERs for Person-Specific FLR
thresholds = Perf.thresholds;

EERs = zeros(SUB_NUM, SUB_NUM);
HTERs = zeros(SUB_NUM, SUB_NUM);
Scores_subs_gen = cell(SUB_NUM, SUB_NUM);
Scores_subs_fake = cell(SUB_NUM, SUB_NUM);

% Assemble genuine (fake)  samples from one subject
for s = 1:SUB_NUM
    disp(strcat('Subject: ', num2str(s)));
    
    samples_data_train_fld_pos = samples_data_subjects_genuine{s};
    samples_data_train_fld_neg = samples_data_subjects_fake{s};

    samples_data_fld_train  =  [samples_data_train_fld_pos; samples_data_train_fld_neg];
    samples_label_fld_train =  [ones(size(samples_data_train_fld_pos, 1), 1); -1*ones(size(samples_data_train_fld_neg, 1), 1)];

    % L2 Normalize feature vectors, features have been normalized before
    samples_data_fld_train = bsxfun(@rdivide, samples_data_fld_train, sqrt(sum(samples_data_fld_train.^2, 2)));

    % ------------- test trained SVM model by trainning data --------------- %
    parfor c = 1:SUB_NUM
        [~,~,scores] = svmpredict(samples_label_fld_train,samples_data_fld_train, PSModels{c});
        
        [EER, ~] = compute_EER(scores(samples_label_fld_train == 1), scores(samples_label_fld_train == -1));
        EERs(s, c) = EER;
        
        Scores_subs_gen{s, c} = scores(samples_label_fld_train == 1);
        Scores_subs_fake{s, c} = scores(samples_label_fld_train == -1);
        
        FAR = sum(scores(samples_label_fld_train == -1) >= thresholds(c))/(sum(samples_label_fld_train == -1));
        FRR = sum(scores(samples_label_fld_train == 1) < thresholds(c))/(sum(samples_label_fld_train == 1));
        HTERs(s, c) = (FAR + FRR)/2;
        
        % disp(strcat('EER: ', num2str(EER), ' HTER: ', num2str(HTERs(s))));
    end
end

% disp(strcat('Mean EER: ', num2str(mean(EERs))));
% disp(strcat('Mean HTER: ', num2str(mean(HTERs))));

Perf.EERs = EERs;
Perf.HTERs = HTERs;
Perf.Scores_gen = Scores_subs_gen;
Perf.Scores_fake = Scores_subs_fake;

save(strcat('PerSpecPerf_crosstest', modelname, '_', method, '.mat'), 'Perf');

end

