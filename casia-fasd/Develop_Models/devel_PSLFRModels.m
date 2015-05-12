function devel_PSLFRModels(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Type, method)
%DEVEL_PSMODELS Summary of this function goes here
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
    if strcmp(TLabels(i), 'PP')
        samples_label_subjects_genuine{s}(i) = 1;
    else
        samples_label_subjects_fake{s}(i) = 1;
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
        samples_gen_data_sub = [samples_gen_data_sub; samples_data_subjects_genuine{s}(ind(int16(length(ind)/3)+1:int16(2*length(ind)/3)), :)];
    end
    samples_data_subjects_genuine{s} = samples_gen_data_sub;
    
    if s <= SUB_NUM_S
        % reassign fake samples for development in source subject domains
        samples_fake_data_sub = [];
        for a = 1:3
            for q = 1:3
                ind = find(strcmp(samples_stype_subjects_fake{s}, SpoofingTypes{a}) & strcmp(samples_qtype_subjects_fake{s}, QualityTypes_Fake{q}));
                samples_fake_data_sub = [samples_fake_data_sub; samples_data_subjects_fake{s}(ind(1+int16(length(ind)/3):int16(2*length(ind)/3)), :)];
            end
        end
        samples_data_subjects_fake{s} = samples_fake_data_sub;
    else
        % reassign fake samples for development in target subject domains
        samples_fake_data_sub = [];
        for a = 1:3
            for q = 1:3
                ind = find(strcmp(samples_stype_subjects_fake{s}, SpoofingTypes{a}) & strcmp(samples_qtype_subjects_fake{s}, QualityTypes_Fake{q}));
                samples_fake_data_sub = [samples_fake_data_sub; samples_data_subjects_fake{s}(ind(1:int16(length(ind)/2)), :)];
            end
        end
        samples_data_subjects_fake{s} = samples_fake_data_sub;
    end
   
end

modelname = '';
for k = 1:length(Feat_Type)
    modelname = strcat(modelname, '_', name{Feat_Type(k)});
end

load(strcat('..\Train_Models\PSModels', modelname, '_', method, '.mat'));

Scores_subs_gen = cell(SUB_NUM, 1);
Scores_subs_fake = cell(SUB_NUM, 1);

% Assemble genuine (fake)  samples from one subject

if matlabpool('size')<=0 %判断并行计算环境是否已然启动
    matlabpool('open','local',8); %若尚未启动，则启动并行环境
else
    disp('Already initialized'); %说明并行环境已经启动。
end

parfor s = 1:SUB_NUM
    disp(strcat('Subject: ', num2str(s)));
    
    samples_data_train_fld_pos = samples_data_subjects_genuine{s};
    samples_data_train_fld_neg = samples_data_subjects_fake{s};

    samples_data_fld_train  =  [samples_data_train_fld_pos; samples_data_train_fld_neg];
    samples_label_fld_train =  [ones(size(samples_data_train_fld_pos, 1), 1); -1*ones(size(samples_data_train_fld_neg, 1), 1)];

    % L2 Normalize feature vectors, has normalized the feature before
    % samples_data_fld_train = bsxfun(@rdivide, samples_data_fld_train, sqrt(sum(samples_data_fld_train.^2, 2)));

    % ------------- test trained SVM model by trainning data --------------- %
    [~,~,scores] = svmpredict(samples_label_fld_train,samples_data_fld_train, PSModels{s});
    
    Scores_subs_gen{s} = scores(samples_label_fld_train == 1);
    Scores_subs_fake{s} = scores(samples_label_fld_train == -1);
end

% disp(strcat('Mean EER: ', num2str(mean(EERs))));

%  Conduct face recognition
load(strcat('..\Train_Models\PSFRModel', modelname, '_', method, '.mat'));

T = FRModel.T;
train_samples_transform = FRModel.samples_transform;

% compute the center for each training group
centers_train_samples = zeros(length(train_samples_transform), SUB_NUM-1);
for i = 1:SUB_NUM
    centers_train_samples(i, :) = mean(train_samples_transform{i}, 2);
end

% Face recognition

FRLabels = cell(1, SUB_NUM);
NewScores_subs_gen = cell(SUB_NUM, 1);
NewScores_subs_fake = cell(SUB_NUM, 1);

for s = 1:SUB_NUM
    
    samples_data_train_fld_pos = samples_data_subjects_genuine{s};
    samples_data_train_fld_neg = samples_data_subjects_fake{s};

    samples_data_fld_train  =  [samples_data_train_fld_pos; samples_data_train_fld_neg];
    num_gen = size(samples_data_train_fld_pos, 1);
    num_fake = size(samples_data_train_fld_neg, 1);
    
    % L2 Normalize feature vectors, has been normalized before
    % samples_data_fld_train = bsxfun(@rdivide, samples_data_fld_train, sqrt(sum(samples_data_fld_train.^2, 2)));

    transform_samples_data_fr_test = T*samples_data_fld_train';
    DistMatrix = pdist2(transform_samples_data_fr_test', centers_train_samples);
    
    % find the maximum scores and the corresponding index 
    [scores_fr_test_min, scores_fr_test_minind] = min(DistMatrix');
    
    FRLabels{s}.scores = scores_fr_test_min';
    FRLabels{s}.inds   = scores_fr_test_minind';
    
    genLabels  = FRLabels{s}.inds(1:num_gen);
    fakeLabels = FRLabels{s}.inds(num_gen+1:num_gen+num_fake);
    
    % Copy the scores of correctly recognized samples to new variables
    NewScores_subs_gen{s} = [NewScores_subs_gen{s}; Scores_subs_gen{s}(genLabels == s)];
    NewScores_subs_fake{s} = [NewScores_subs_fake{s}; Scores_subs_fake{s}(fakeLabels == s)];    
    
    % As for the incorrectly recognized samples, should obtain the new scores for them using the corresponding face recognizers
    % compute new scores for genuine samples first
    ind = find(genLabels ~= s);
    for i = 1:length(ind)        
        sample = samples_data_train_fld_pos(ind(i));
        % conduct face recognition for the samples
        [~,~,score] = svmpredict(1,sample, PSModels{genLabels(ind(i))});
        NewScores_subs_gen{genLabels(ind(i))} = [NewScores_subs_gen{genLabels(ind(i))}; score];
    end
    
    % compute new scores for fake samples secondly
    ind = find(fakeLabels ~= s);
    for i = 1:sum(fakeLabels ~= s)
        sample = samples_data_train_fld_neg(ind(i));
        % conduct face recognition for the samples
        [~,~,score] = svmpredict(-1,sample, PSModels{fakeLabels(ind(i))});
        NewScores_subs_fake{fakeLabels(ind(i))} = [NewScores_subs_fake{fakeLabels(ind(i))}; score];
    end
        
end

EERs = zeros(SUB_NUM, 1);
thresholds = zeros(SUB_NUM, 1);

for s = 1:SUB_NUM

    [EER, thresholds(s)] = compute_EER(NewScores_subs_gen{s}, NewScores_subs_fake{s});
    EERs(s) = EER;
    
    FAR = sum(NewScores_subs_fake{s} >= thresholds(s))/size(NewScores_subs_fake{s}, 1);
    FRR = sum(NewScores_subs_gen{s} < thresholds(s))/size(NewScores_subs_gen{s}, 1);
    HTERs(s) = (FAR + FRR)/2;    
    
    disp(strcat('EER: ', num2str(EER), ' HTER: ', num2str(HTERs(s))));
end

Perf.EERs = EERs;
Perf.thresholds = thresholds;
save(strcat('PerSpecLFRPerf_devel', modelname, '_', method, '.mat'), 'Perf');

end

