function test_FRModels_fusion(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feat_Type)
%TEST_FRMODELS Summary of this function goes here
%   Detailed explanation goes here
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
    elseif strcmp(PNLabels(i), 'N')  % if the sample is from source subject, then add it into training set
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
        
    elseif strcmp(PNLabels(i), 'N')
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
        samples_gen_data_sub = [samples_gen_data_sub; samples_data_subjects_genuine{s}(ind(1+int16(2*length(ind)/3):end), :)];
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
                    if length(ind) == 0
                        continue;
                    end
                    samples_fake_data_sub = [samples_fake_data_sub; samples_data_subjects_fake{s}(ind(1+int16(2*length(ind)/3):end), :)];
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
        samples_gen_data_sub = [samples_gen_data_sub; samples_data_subjects_genuine{s}(ind(1+int16(2*length(ind)/3):end), :)];
    end
    samples_data_subjects_genuine{s} = samples_gen_data_sub;
    
    % reassign fake samples: the first 1/2
    samples_fake_data_sub = [];
    for b = 1:2
        for m = 1:2
            for a = 1:3
                for f = 1:2
                    ind = find(strcmp(samples_btype_subjects_fake{s}, BTypes_fake{b}) & strcmp(samples_mtype_subjects_fake{s}, MTypes_fake{m}) & ...
                                      strcmp(samples_atype_subjects_fake{s}, ATypes_fake{a}) & strcmp(samples_ftype_subjects_fake{s}, FTypes_fake{f}));
                    samples_fake_data_sub = [samples_fake_data_sub; samples_data_subjects_fake{s}(ind(1+int16(length(ind)/2):end), :)];
                end
            end
        end
    end
    samples_data_subjects_fake{s} = samples_fake_data_sub;    
end

%  Conduct face recognition
modelname = '';
for k = 1:length(Feat_Type)
    modelname = strcat(modelname, '_', name{Feat_Type(k)});
end

load(strcat('..\Train_Models\FRModel', modelname, '.mat'));

T = FRModel.T;
train_samples_transform = FRModel.samples_transform;

% compute the center for each training group
centers_train_samples = zeros(length(train_samples_transform), SUB_NUM-1);
for i = 1:SUB_NUM
    centers_train_samples(i, :) = mean(train_samples_transform{i}, 2);
end

% Face recognition

FRLabels = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    
    samples_data_train_fld_pos = samples_data_subjects_genuine{s};
    samples_data_train_fld_neg = samples_data_subjects_fake{s};

    samples_data_fld_train  =  [samples_data_train_fld_pos; samples_data_train_fld_neg];
    
    % L2 Normalize feature vectors, normalized before
    % samples_data_fld_train = bsxfun(@rdivide, samples_data_fld_train, sqrt(sum(samples_data_fld_train.^2, 2)));

    transform_samples_data_fr_test = T*samples_data_fld_train';
    
    
%     if s ~= 37
%         centers_train_samples_rep = centers_train_samples;
%         centers_train_samples_rep(37, :) = 0;
%         DistMatrix = pdist2(transform_samples_data_fr_test', centers_train_samples_rep);
%     else
        DistMatrix = pdist2(transform_samples_data_fr_test', centers_train_samples);        
%     end
    
    % find the maximum scores and the corresponding index 
    [scores_fr_test_min, scores_fr_test_minind] = min(DistMatrix');
    
    FRLabels{s}.scores = scores_fr_test_min';
    FRLabels{s}.inds   = scores_fr_test_minind';
end

save(strcat('FRPerf_test_', modelname, '.mat'), 'FRLabels');

end

