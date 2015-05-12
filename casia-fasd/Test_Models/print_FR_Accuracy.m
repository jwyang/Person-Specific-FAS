% load face recognition results
FR_Results_train = textread('D:/Projects_Face_AntiSpoofing/Datasets/CASIA-FASD/train_release/FR_Results.txt');
FR_Results_test = textread('D:/Projects_Face_AntiSpoofing/Datasets/CASIA-FASD/test_release/FR_Results.txt');

QualityType_Gen = strcat('L', 'P');
QualityType_Fake = strcat('L', 'N');

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
FR_Results = [FR_Results_train; FR_Results_test];

idx_test_genuine = cell(1, SUB_NUM);
idx_test_fake = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    idx_test_genuine{s} = zeros(1, length(SubIDs));
    idx_test_fake{s} = zeros(1, length(SubIDs));
end

for i = 1:length(SubIDs)
    s = SubIDs(i);
    if strcmp(QLabels(i), QualityType_Gen)
        idx_test_genuine{s}(i) = 1;
    elseif strcmp(QLabels(i), QualityType_Fake)
        idx_test_fake{s}(i) = 1;    
    end
end

fr_results_subjects_genuine = cell(1, SUB_NUM);         % save real and virtual fake samples for training
fr_results_subjects_fake = cell(1, SUB_NUM);     % save genuine samples for training

samples_qtype_subjects_genuine = cell(1, SUB_NUM);
samples_qtype_subjects_fake = cell(1, SUB_NUM);
samples_stype_subjects_genuine = cell(1, SUB_NUM);
samples_stype_subjects_fake = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    fr_results_subjects_genuine{s} = zeros(sum(idx_test_genuine{s}), 1);
    fr_results_subjects_fake{s} = zeros(sum(idx_test_fake{s}), 1);    
    
    samples_qtype_subjects_genuine{s} = cell(sum(idx_test_genuine{s}), 1);
    samples_stype_subjects_genuine{s} = cell(sum(idx_test_genuine{s}), 1);

    samples_qtype_subjects_fake{s} = cell(sum(idx_test_fake{s}), 1);
    samples_stype_subjects_fake{s} = cell(sum(idx_test_fake{s}), 1);
    
end


sample_sub_id_genuine = ones(1, SUB_NUM);
sample_sub_id_fake = ones(1, SUB_NUM);

for i = 1:length(SubIDs)
    s = SubIDs(i);
    if strcmp(QLabels(i), QualityType_Gen)
        fr_results_subjects_genuine{s}(sample_sub_id_genuine(s), :) = FR_Results(i, 1);
        samples_stype_subjects_genuine{s}(sample_sub_id_genuine(s)) = TLabels(i);
        samples_qtype_subjects_genuine{s}(sample_sub_id_genuine(s)) = QLabels(i);
        sample_sub_id_genuine(s) = sample_sub_id_genuine(s) + 1;
        
    elseif strcmp(QLabels(i), QualityType_Fake)
        fr_results_subjects_fake{s}(sample_sub_id_fake(s)) = FR_Results(i, 1);
        samples_stype_subjects_fake{s}(sample_sub_id_fake(s)) = TLabels(i);
        samples_qtype_subjects_fake{s}(sample_sub_id_fake(s)) = QLabels(i);
        sample_sub_id_fake(s) = sample_sub_id_fake(s) + 1;
    end    
end

% Assign samples for training
SpoofingTypes = {'CPN' 'WPN' 'VN'};

for s = 1:SUB_NUM
    % reassign genuine samples
    % samples_gen_data_sub = [];
    % for q = 1:length(QualityType_Gen)
    %    ind = find(strcmp(samples_qtype_subjects_genuine{s}, QualityType_Gen));
    %    samples_gen_data_sub = [samples_gen_data_sub; samples_data_subjects_genuine{s}(1:int16(length(ind)/2), :)];  % the first half samples
    % end
    ind = find(strcmp(samples_qtype_subjects_genuine{s}, QualityType_Gen));
    fr_results_subjects_genuine{s} = fr_results_subjects_genuine{s}(1+int16(length(ind)/2):end, :);  % the first half samples
    
    
    % reassign fake samples for development in target subject domains
    samples_fake_data_sub = [];
    % for q = 1:length(QualityType_Fake)
        for a = 1:length(SpoofingTypes)        
            ind = find(strcmp(samples_stype_subjects_fake{s}, SpoofingTypes{a}) & strcmp(samples_qtype_subjects_fake{s}, QualityType_Fake));
            samples_fake_data_sub = [samples_fake_data_sub; fr_results_subjects_fake{s}(ind(1+int16(length(ind)/2):end), :)];
        end
    % end
    fr_results_subjects_fake{s} = samples_fake_data_sub;
    
end

% save fr results into driver
save('fr_results_subjects_genuine.mat', 'fr_results_subjects_genuine');
save('fr_results_subjects_fake.mat', 'fr_results_subjects_fake');

num_samples = 0;
num_hit = 0;
for i = 1:SUB_NUM_S
    s = i;
    num_samples = num_samples + size(fr_results_subjects_genuine{s}, 1);
    num_samples = num_samples + size(fr_results_subjects_fake{s}, 1);
    
    num_hit = num_hit + sum(fr_results_subjects_genuine{s} == (s));
    num_hit = num_hit + sum(fr_results_subjects_fake{s} == (s));
end
accuracy_S = num_hit / num_samples;

num_samples = 0;
num_hit = 0;
for i = 1:SUB_NUM_T
    s = i + 20;
    num_samples = num_samples + size(fr_results_subjects_genuine{s}, 1);
    num_samples = num_samples + size(fr_results_subjects_fake{s}, 1);
    
    num_hit = num_hit + sum(fr_results_subjects_genuine{s} == (s));
    num_hit = num_hit + sum(fr_results_subjects_fake{s} == (s));
end
accuracy_T = num_hit / num_samples;

num_samples = 0;
num_hit = 0;
for s = 1:SUB_NUM
    num_samples = num_samples + size(fr_results_subjects_genuine{s}, 1);
    num_samples = num_samples + size(fr_results_subjects_fake{s}, 1);
    
    num_hit = num_hit + sum(fr_results_subjects_genuine{s} == (s));
    num_hit = num_hit + sum(fr_results_subjects_fake{s} == (s));
end

accuracy = num_hit / num_samples;
