function train_PSModels_withSubNum(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feat_Type, method)
%TRAIN_PSMODELS Summary of this function goes here
%   Function: train person-specific face anti-spoofing classifiers for both source subjects and target subjects
%   Detailed explanation goes here
%   Input:
%        Feats_train_SL: feature set of first 20 subjects
%        Labels_train_SL: label set of first 20 subjects
%        Feats_test_SL: feature set of remaining 30 subjects
%        Labels_test_SL: feature set of remaning 30 subjects
%        Feat_Type: type of feature used for face anti-spoofing
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

% concatenate train and test information
SubIDs = [SubIDs_train, SubIDs_devel, SubIDs_test];
PNLabels = [PNLabels_train, PNLabels_devel, PNLabels_test];
BLabels = [BLabels_train, BLabels_devel, BLabels_test];
MLabels = [MLabels_train, MLabels_devel, MLabels_test];
ALabels = [ALabels_train, ALabels_devel, ALabels_test];
FLabels = [FLabels_train, FLabels_devel, FLabels_test];
Feats   = [Feats_train, Feats_devel, Feats_test];

clientID = unique(SubIDs);

clientID_source  = unique(SubIDs_train);
clientID_target = unique([SubIDs_devel, SubIDs_test]);

SUB_NUM_S = length(clientID_source);
SUB_NUM_T = length(clientID_target);
SUB_NUM = SUB_NUM_S + SUB_NUM_T;

samples_label_subject = zeros(length(SubIDs), 1);

samples_label_subjects_genuine = cell(1, SUB_NUM);
samples_label_subjects_fake = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    samples_label_subjects_genuine{s} = uint8(samples_label_subject);
    samples_label_subjects_fake{s} = uint8(samples_label_subject);    
end

for i = 1:length(SubIDs)
    s_rank = find(SubIDs(i)==clientID);
    s = s_rank(1);
    if strcmp(PNLabels(i), 'P') % && strcmp(ALabels(i), 'IpN') && strcmp(MLabels(i), 'FixN') && strcmp(FLabels(i), 'PhN')
        samples_label_subjects_genuine{s}(i) = 1;
    elseif strcmp(PNLabels(i), 'N') && sum(SubIDs(i) == clientID_source) > 0   % if the sample is from source subject, then add it into training set
        samples_label_subjects_fake{s}(i) = 1;        
    end
end

% global samples_data_subjects_genuine;
% global samples_data_subjects_fake;

samples_data_subjects_genuine = cell(1, SUB_NUM);         % save real and virtual fake samples for training
samples_data_subjects_fake = cell(1, SUB_NUM);     % save genuine samples for training

samples_btype_subjects_genuine = cell(1, SUB_NUM);

samples_btype_subjects_fake = cell(1, SUB_NUM);
samples_mtype_subjects_fake= cell(1, SUB_NUM);
samples_atype_subjects_fake = cell(1, SUB_NUM);
samples_ftype_subjects_fake = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    samples_data_subjects_genuine{s} = zeros(sum(samples_label_subjects_genuine{s}), dims(Feat_Type));
    samples_data_subjects_fake{s} = zeros(sum(samples_label_subjects_fake{s}), dims(Feat_Type));    
    
    samples_btype_subjects_genuine{s} = cell(sum(samples_label_subjects_genuine{s}), 1);
    
    samples_btype_subjects_fake{s} = cell(sum(samples_label_subjects_fake{s}), 1);
    samples_mtype_subjects_fake{s} = cell(sum(samples_label_subjects_fake{s}), 1);
    samples_atype_subjects_fake{s} = cell(sum(samples_label_subjects_fake{s}), 1);
    samples_ftype_subjects_fake{s} = cell(sum(samples_label_subjects_fake{s}), 1);
    
end

sample_sub_id_genuine = ones(1, SUB_NUM);
sample_sub_id_fake = ones(1, SUB_NUM);

for i = 1:length(SubIDs)
    s_rank = find(SubIDs(i)==clientID);
    s = s_rank(1);
    if strcmp(PNLabels(i), 'P') % && strcmp(ALabels(i), 'IpN')  && strcmp(MLabels(i), 'FixN') && strcmp(FLabels(i), 'PhN')
        if Feat_Type == 1
            samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), :) = Feats{i}.MsLBP{1};
        elseif Feat_Type == 2
            samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), :) = Feats{i}.LBP{1};
        elseif Feat_Type == 3
            samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), :) = Feats{i}.HOG{1};
        elseif Feat_Type == 4
            samples_data_subjects_genuine{s}(sample_sub_id_genuine(s), :) = Feats{i}.LPQ{1};
        end
        samples_btype_subjects_genuine{s}(sample_sub_id_genuine(s)) = BLabels(i);
        sample_sub_id_genuine(s) = sample_sub_id_genuine(s) + 1;        
        
    elseif strcmp(PNLabels(i), 'N') && sum(SubIDs(i) == clientID_source) > 0
        if Feat_Type == 1
            samples_data_subjects_fake{s}(sample_sub_id_fake(s), :) = Feats{i}.MsLBP{1};
        elseif Feat_Type == 2
            samples_data_subjects_fake{s}(sample_sub_id_fake(s), :) = Feats{i}.LBP{1};
        elseif Feat_Type == 3
            samples_data_subjects_fake{s}(sample_sub_id_fake(s), :) = Feats{i}.HOG{1};
        elseif Feat_Type == 4
            samples_data_subjects_fake{s}(sample_sub_id_fake(s), :) = Feats{i}.LPQ{1};
        end        
        samples_btype_subjects_fake{s}(sample_sub_id_fake(s)) = BLabels(i);
        samples_mtype_subjects_fake{s}(sample_sub_id_fake(s)) = MLabels(i);
        samples_atype_subjects_fake{s}(sample_sub_id_fake(s)) = ALabels(i);
        samples_ftype_subjects_fake{s}(sample_sub_id_fake(s)) = FLabels(i);
        sample_sub_id_fake(s) = sample_sub_id_fake(s) + 1;
    end    
end

% Assign samples for training
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


if matlabpool('size')<=0 %判断并行计算环境是否已然启动
    matlabpool('open','local',8); %若尚未启动，则启动并行环境
else
    disp('Already initialized'); %说明并行环境已经启动。
end

% Load virtual fake features of target subjects for training
VirtualFeatures = cell(2, 1);

for b = 1:2
    VirtualFeatures{b} = load(strcat('..\Domain_Adaptation\SynthFeatures_AllTSubs_withSubNum_', BTypes_fake{b}, '_', name{Feat_Type}, '_', method));
end

% Train SVM classifier for person-specific face anti-spoofing
PSModels = cell(SUB_NUM, SUB_NUM_S);

for SubNum = 1:SUB_NUM_S
    for i = 1:SUB_NUM_T
        subid = clientID_target(i);
        s_rank = find(subid == clientID);
        s = s_rank(1);
        
        samples_data = VirtualFeatures{1}.SynthFeatures_AllTSubs{i, SubNum};
        samples_data = [samples_data; VirtualFeatures{2}.SynthFeatures_AllTSubs{i, SubNum}];
        
        samples_data_subjects_fake{s} = samples_data;
    end
    

    
    parfor s = 1:SUB_NUM
        samples_data_train_fld_pos = samples_data_subjects_genuine{s};
        samples_data_train_fld_neg = samples_data_subjects_fake{s};
        PSModels{s, SubNum} = trainFLDClassifierforsubject(samples_data_train_fld_pos, samples_data_train_fld_neg);
    end
    
end

save(strcat('PSModels_withSubNum_', name{Feat_Type}, '_', method, '.mat'), 'PSModels', '-v7.3');

end


function model = trainFLDClassifierforsubject(samples_data_train_fld_pos, samples_data_train_fld_neg)
% 
% global samples_data_subjects_genuine;
% global samples_data_subjects_fake;


% samples_data_train_fld_pos = samples_data_subjects_genuine{SubID};
% samples_data_train_fld_neg = samples_data_subjects_fake{SubID};

samples_data_fld_train  =  [samples_data_train_fld_pos; samples_data_train_fld_neg];
samples_label_fld_train =  [ones(size(samples_data_train_fld_pos, 1), 1); -1*ones(size(samples_data_train_fld_neg, 1), 1)];

% L2 Normalize feature vectors
samples_data_fld_train = bsxfun(@rdivide, samples_data_fld_train, sqrt(sum(samples_data_fld_train.^2, 2)));

% svm_param = ['-t 0 -c 0.5'];
for t = 0
    for c = 100
        svm_param = strcat('-t ',num2str(t), ' -c ', num2str(c), ' -g 0.5 -h 0');
        
        % ---------------- train SVM model ---------------- %
        model = svmtrain(samples_label_fld_train, samples_data_fld_train, svm_param);
        
        % ------------- test trained SVM model by trainning data --------------- %
        svmpredict(samples_label_fld_train,samples_data_fld_train,model);
    end
end

clear samples_data_fld_train；
clear samples_label_fld_train;

end


