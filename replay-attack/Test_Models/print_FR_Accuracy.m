% load face recognition results
FR_Results_train = textread('D:/Projects_Face_AntiSpoofing/Datasets/Replay-Attack/train/FR_Results.txt');
FR_Results_devel = textread('D:/Projects_Face_AntiSpoofing/Datasets/Replay-Attack/develop/FR_Results.txt');
FR_Results_test = textread('D:/Projects_Face_AntiSpoofing/Datasets/Replay-Attack/test/FR_Results.txt');

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

FR_Results = [FR_Results_train; FR_Results_devel; FR_Results_test];
SubIDs = [SubIDs_train, SubIDs_devel, SubIDs_test];
PNLabels = [PNLabels_train, PNLabels_devel, PNLabels_test];
BLabels = [BLabels_train, BLabels_devel, BLabels_test];
MLabels = [MLabels_train, MLabels_devel, MLabels_test];
ALabels = [ALabels_train, ALabels_devel, ALabels_test];
FLabels = [FLabels_train, FLabels_devel, FLabels_test];

clientID = unique(SubIDs);
clientID_source  = unique(SubIDs_train);
clientID_target = unique([SubIDs_devel, SubIDs_test]);

SUB_NUM_S = length(clientID_source);
SUB_NUM_T = length(clientID_target);
SUB_NUM = SUB_NUM_S + SUB_NUM_T;

idx_test_genuine = cell(1, SUB_NUM);
idx_test_fake = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    idx_test_genuine{s} = zeros(1, length(SubIDs));
    idx_test_fake{s} = zeros(1, length(SubIDs));
end

for i = 1:length(SubIDs)
    s_rank = find(SubIDs(i)==clientID);
    s = s_rank(1);
    if strcmp(PNLabels(i), 'P') % && strcmp(ALabels(i), 'IpN') && strcmp(MLabels(i), 'FixN') && strcmp(FLabels(i), 'PhN')
        idx_test_genuine{s}(i) = 1;
    elseif strcmp(MLabels(i), 'HandN')  % if the sample is from source subject, then add it into training set
        idx_test_fake{s}(i) = 1;    
    end
end

fr_results_subjects_genuine = cell(1, SUB_NUM);         % save real and virtual fake samples for training
fr_results_subjects_fake = cell(1, SUB_NUM);     % save genuine samples for training

samples_btype_subjects_genuine = cell(1, SUB_NUM);
samples_btype_subjects_fake = cell(1, SUB_NUM);
samples_mtype_subjects_fake= cell(1, SUB_NUM);
samples_atype_subjects_fake = cell(1, SUB_NUM);
samples_ftype_subjects_fake = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    fr_results_subjects_genuine{s} = zeros(sum(idx_test_genuine{s}), 1);
    fr_results_subjects_fake{s} = zeros(sum(idx_test_fake{s}), 1);    
    
    samples_btype_subjects_genuine{s} = cell(sum(idx_test_genuine{s}), 1);
    
    samples_btype_subjects_fake{s} = cell(sum(idx_test_fake{s}), 1);
    samples_mtype_subjects_fake{s} = cell(sum(idx_test_fake{s}), 1);
    samples_atype_subjects_fake{s} = cell(sum(idx_test_fake{s}), 1);
    samples_ftype_subjects_fake{s} = cell(sum(idx_test_fake{s}), 1);
    
end


sample_sub_id_genuine = ones(1, SUB_NUM);
sample_sub_id_fake = ones(1, SUB_NUM);

for i = 1:length(SubIDs)
    s_rank = find(SubIDs(i)==clientID);
    s = s_rank(1);
    if strcmp(PNLabels(i), 'P') % && strcmp(ALabels(i), 'IpN')  && strcmp(MLabels(i), 'FixN') && strcmp(FLabels(i), 'PhN')
        fr_results_subjects_genuine{s}(sample_sub_id_genuine(s), :) = FR_Results(i, 1);
        samples_btype_subjects_genuine{s}(sample_sub_id_genuine(s)) = BLabels(i);
        sample_sub_id_genuine(s) = sample_sub_id_genuine(s) + 1;
        
    elseif strcmp(MLabels(i), 'HandN')
        fr_results_subjects_fake{s}(sample_sub_id_fake(s)) = FR_Results(i, 1);
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
    fr_result_genuine_data_sub = [];
    for b = 1:2
        ind = find(strcmp(samples_btype_subjects_genuine{s}, BTypes_genuine{b}));
        fr_result_genuine_data_sub = [fr_result_genuine_data_sub; fr_results_subjects_genuine{s}(ind(1+int16(length(ind)/2):end))];
    end
    fr_results_subjects_genuine{s} = fr_result_genuine_data_sub;
    
    % reassign fake samples
    fr_result_fake_data_sub = [];
    for b = 1:2
        for m = 1:2
            for a = 1:3
                for f = 1:2
                    ind = find(strcmp(samples_btype_subjects_fake{s}, BTypes_fake{b}) & strcmp(samples_mtype_subjects_fake{s}, MTypes_fake{m}) & ...
                                      strcmp(samples_atype_subjects_fake{s}, ATypes_fake{a}) & strcmp(samples_ftype_subjects_fake{s}, FTypes_fake{f}));
                    if isempty(ind)
                        continue;
                    end
                    fr_result_fake_data_sub = [fr_result_fake_data_sub; fr_results_subjects_fake{s}(ind(1+int16(length(ind)/2):end), :)];
                end
            end
        end
    end
    fr_results_subjects_fake{s} = fr_result_fake_data_sub;
end


for i = 1:SUB_NUM_T  % and then assign for target subjects
    subid = clientID_target(i);
    s_rank = find(subid == clientID);
    s = s_rank(1);
    
    % reassign genuine samples: the first 1/3
    fr_result_genuine_data_sub = [];
    for b = 1:2
        ind = find(strcmp(samples_btype_subjects_genuine{s}, BTypes_genuine{b}));
        fr_result_genuine_data_sub = [fr_result_genuine_data_sub; fr_results_subjects_genuine{s}(ind(1+int16(length(ind)/2):end))];
    end
    fr_results_subjects_genuine{s} = fr_result_genuine_data_sub;
    
    % reassign fake samples
    fr_result_fake_data_sub = [];
    for b = 1:2
        for m = 1:2
            for a = 1:3
                for f = 1:2
                    ind = find(strcmp(samples_btype_subjects_fake{s}, BTypes_fake{b}) & strcmp(samples_mtype_subjects_fake{s}, MTypes_fake{m}) & ...
                                      strcmp(samples_atype_subjects_fake{s}, ATypes_fake{a}) & strcmp(samples_ftype_subjects_fake{s}, FTypes_fake{f}));
                    if isempty(ind)
                        continue;
                    end
                    fr_result_fake_data_sub = [fr_result_fake_data_sub; fr_results_subjects_fake{s}(ind(1+int16(length(ind)/2):end), :)];
                end
            end
        end
    end
    fr_results_subjects_fake{s} = fr_result_fake_data_sub;
end

% save fr results into driver
save('fr_results_subjects_genuine.mat', 'fr_results_subjects_genuine');
save('fr_results_subjects_fake.mat', 'fr_results_subjects_fake');

num_samples = 0;
num_hit = 0;
for i = 1:SUB_NUM_S
    subid = clientID_source(i);
    s_rank = find(subid == clientID);
    s = s_rank(1);
    
    num_samples = num_samples + size(fr_results_subjects_genuine{s}, 1);
    num_samples = num_samples + size(fr_results_subjects_fake{s}, 1);
    
    num_hit = num_hit + sum(fr_results_subjects_genuine{s} == clientID(s));
    num_hit = num_hit + sum(fr_results_subjects_fake{s} == clientID(s));
end
accuracy = num_hit / num_samples;

num_samples = 0;
num_hit = 0;
for i = 1:SUB_NUM_T
    subid = clientID_target(i);
    s_rank = find(subid == clientID);
    s = s_rank(1);
    
    num_samples = num_samples + size(fr_results_subjects_genuine{s}, 1);
    num_samples = num_samples + size(fr_results_subjects_fake{s}, 1);
    
    num_hit = num_hit + sum(fr_results_subjects_genuine{s} == clientID(s));
    num_hit = num_hit + sum(fr_results_subjects_fake{s} == clientID(s));
end
accuracy = num_hit / num_samples;

num_samples = 0;
num_hit = 0;
for s = 1:SUB_NUM
    num_samples = num_samples + size(fr_results_subjects_genuine{s}, 1);
    num_samples = num_samples + size(fr_results_subjects_fake{s}, 1);
    
    num_hit = num_hit + sum(fr_results_subjects_genuine{s} == clientID(s));
    num_hit = num_hit + sum(fr_results_subjects_fake{s} == clientID(s));
end

accuracy = num_hit / num_samples;
