function draw_PComponentsSamples(Feats_train_SL, Labels_train_SL, Feats_test_SL, Labels_test_SL, Feat_Type, method)
%DRAW_PCOMPONENTSSAMPLES Summary of this function goes here
%   Draw principle components of samples in a sub-space
%   Detailed explanation goes here

% Step 1: Organize features
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

        if strcmp(TLabels(i), 'PP') % && strcmp(QLabels(i), 'MP')
            samples_label_subjects_genuine{s}(i) = 1;
        else % if strcmp(QLabels(i), 'MN')
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
        if strcmp(TLabels(i), 'PP') % && strcmp(QLabels(i), 'MP')
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
        else % if strcmp(QLabels(i), 'MN') % && strcmp(TLabels(i), 'VN')
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
QualityTypes_Fake = {'LN' 'MN' 'HN'};
SpoofingTypes = {'CPN' 'WPN' 'VN'};

% for s = 1:SUB_NUM
%     % reassign genuine samples
%     samples_gen_data_sub = [];
%     for q = 1:3
%         ind = find(strcmp(samples_qtype_subjects_genuine{s}, QualityTypes_Genuine{q}));
%         samples_gen_data_sub = [samples_gen_data_sub; samples_data_subjects_genuine{s}(ind(1:int16(length(ind)/3)), :)];
%     end
%     samples_data_subjects_genuine{s} = samples_gen_data_sub;
% 
% %     if s > SUB_NUM_S
% %         continue;
% %     end
% 
%     % reassign fake samples
%     samples_fake_data_sub = [];
%     for a = 1:3
%         for q = 1:3
%             ind = find(strcmp(samples_stype_subjects_fake{s}, SpoofingTypes{a}) & strcmp(samples_qtype_subjects_fake{s}, QualityTypes_Fake{q}));
%             samples_fake_data_sub = [samples_fake_data_sub; samples_data_subjects_fake{s}(ind(1+int16(length(ind)/2):end), :)];
%         end
%     end
%     samples_data_subjects_fake{s} = samples_fake_data_sub;
% end


% Conduct Principle Component Analysis given samples from all subjects

% Assemble all samples
samples_data_allsubs = [];
for s = 1:SUB_NUM
    samples_data_onesub = [samples_data_subjects_genuine{s}; samples_data_subjects_fake{s}];
    samples_data_allsubs = [samples_data_allsubs; samples_data_onesub];
end


modelname = '';
for k = 1:length(Feat_Type)
    modelname = strcat(modelname, '_', name{Feat_Type(k)});
end

if ~exist(strcat('ProjMat', modelname, '.mat'))   
    % PCA 
    ProjMat = pca(samples_data_allsubs);
    save(strcat('ProjMat', modelname, '.mat'), 'ProjMat');
else
    load(strcat('ProjMat', modelname, '.mat'));
end

% Load virtual fake features of target subjects for training
VirtualFeatures = cell(3, length(Feat_Type));

for q = 1:3
    for k = 1:length(Feat_Type)
        VirtualFeatures{q, k} = load(strcat('..\Domain_Adaptation\SynthFeatures_AllTSubs_noOutlier_', QualityTypes_Fake{q}, '_', name{Feat_Type(k)}, '_', method));
    end
end

vsamples_data_subjects_fake = cell(SUB_NUM, 1);

for s = SUB_NUM_S+1:SUB_NUM
    samples_data = VirtualFeatures{1}.SynthFeatures_AllTSubs{s-SUB_NUM_S};
    samples_data = [samples_data; VirtualFeatures{2}.SynthFeatures_AllTSubs{s-SUB_NUM_S}];
    
    % Translate high quality samples according to the distributions of low and normal quality samples
%     center_lq = mean(VirtualFeatures{1}.SynthFeatures_AllTSubs{s-SUB_NUM_S}, 1);
%     center_mq = mean(VirtualFeatures{2}.SynthFeatures_AllTSubs{s-SUB_NUM_S}, 1);
%     
%     center_hq_lq = center_lq + CS_HN_LN;
%     center_hq_mq = center_mq + CS_HN_MN;
%     
%     center_hq = (center_hq_lq + center_hq_mq)/2;    
%     center_hq_f = mean(VirtualFeatures{3}.SynthFeatures_AllTSubs{s-SUB_NUM_S}, 1);
%     
%     VirtualFeatures{3}.SynthFeatures_AllTSubs{s-SUB_NUM_S} = bsxfun(@plus, VirtualFeatures{3}.SynthFeatures_AllTSubs{s-SUB_NUM_S}, (center_hq - center_hq_f));
     samples_data = [samples_data; VirtualFeatures{3}.SynthFeatures_AllTSubs{s-SUB_NUM_S}];
%     
%     samples_data(samples_data < 0) = 0;
    
    vsamples_data_subjects_fake{s} = bsxfun(@rdivide, samples_data, sqrt(sum(samples_data.^2, 2)));
end

%{
for s = SUB_NUM_S+1:SUB_NUM

    samples_data_q1 = [];
    for k = 1:length(Feat_Type)
        data = VirtualFeatures{1, k}.SynthFeatures_AllTSubs{s-SUB_NUM_S};
        if isempty(data)
            break;
        end
        data = bsxfun(@rdivide, data, sqrt(sum(data.^2, 2)));  
        if k == 1
            samples_data_q1 = data;
            continue;
        end
        if size(samples_data_q1, 1) > size(data, 1)     
        samples_data_q1 = [samples_data_q1(1:size(data, 1), :), data];
        elseif size(samples_data_q1, 1) <= size(data, 1)
        samples_data_q1 = [samples_data_q1, data(1:size(samples_data_q1, 1), :)];            
        end
    end
    
    samples_data_q2 = [];
    for k = 1:length(Feat_Type)
        data = VirtualFeatures{2, k}.SynthFeatures_AllTSubs{s-SUB_NUM_S};
        if isempty(data)
            break;
        end
        data = bsxfun(@rdivide, data, sqrt(sum(data.^2, 2)));     
        if k == 1
            samples_data_q2 = data;
            continue;
        end
        if size(samples_data_q2, 1) > size(data, 1)     
        samples_data_q2 = [samples_data_q2(1:size(data, 1), :), data];
        elseif size(samples_data_q2, 1) <= size(data, 1)
        samples_data_q2 = [samples_data_q2, data(1:size(samples_data_q2, 1), :)];            
        end
    end    

    samples_data_q3 = [];
    for k = 1:length(Feat_Type)
        data = VirtualFeatures{3, k}.SynthFeatures_AllTSubs{s-SUB_NUM_S};
        if isempty(data)
            break;
        end
        data = bsxfun(@rdivide, data, sqrt(sum(data.^2, 2)));      
        if k == 1
            samples_data_q3 = data;
            continue;
        end
        if size(samples_data_q3, 1) > size(data, 1)
            samples_data_q3 = [samples_data_q3(1:size(data, 1), :), data];
        elseif size(samples_data_q3, 1) <= size(data, 1)
            samples_data_q3 = [samples_data_q3, data(1:size(samples_data_q3, 1), :)];
        end
    end
%     
    
    vsamples_data_subjects_fake{s} = [samples_data_q1; samples_data_q2; samples_data_q3];
    
    % vsamples_data_subjects_fake{s}(vsamples_data_subjects_fake{s} < 0) = 0;
end
%}
close all;
figure, hold;

I = eye(sum(dims(Feat_Type)), sum(dims(Feat_Type)));

cdists = zeros(SUB_NUM_T, 1);
stddevs = zeros(SUB_NUM_T, 1);
covs = zeros(SUB_NUM_T, 1);

% Draw real samples in the 3-D sub-space
for i = 21:SUB_NUM

    s = i;
    
    samples_data_gen = samples_data_subjects_genuine{s};
    samples_data_fake = samples_data_subjects_fake{s};
    vsamples_data_fake = vsamples_data_subjects_fake{s};
    
    cdists(i-SUB_NUM_S) = sqrt(sum((mean(samples_data_fake, 1)-mean(vsamples_data_fake, 1)).^2));
    
    covmat = cov(vsamples_data_fake);
    covs(i - SUB_NUM_S) = sum((covmat(I == 1)));
    stddevs(i - SUB_NUM_S) = sum(sqrt(covmat(I == 1)));
    
    % Project the smaples into 3-D sub-space;
%     proj_samples_data_gen = samples_data_gen*ProjMat(:, 1:3);
%     proj_samples_data_fake = samples_data_fake*ProjMat(:, 1:3);
%     proj_vsamples_data_fake = vsamples_data_fake(1:end, :)*ProjMat(:, 1:3);
%     
%     % plot the samples in sub-space
%     figure, hold on;
%     % plot the real genuine and fake samples from one subject
%     plot3(proj_samples_data_gen(:, 1), proj_samples_data_gen(:, 2), proj_samples_data_gen(:, 3), '.r');
%     plot3(proj_samples_data_fake(:, 1), proj_samples_data_fake(:, 2), proj_samples_data_fake(:, 3), '.g');   
%     
%     % plot the virtual fake samples from one subject
%     plot3(proj_vsamples_data_fake(:, 1), proj_vsamples_data_fake(:, 2), proj_vsamples_data_fake(:, 3), '.b');       
%     close;
end

stat.cdists = cdists;
stat.stddevs = stddevs;
stat.covs = covs;

save(strcat('Stat_', name{Feat_Type}, '_', method), 'stat', '-v7.3');

end

