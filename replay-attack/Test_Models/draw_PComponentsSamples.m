function draw_PComponentsSamples(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feat_Type, method)
%DRAW_PCOMPONENTSSAMPLES Summary of this function goes here
%   Draw principle components of samples in a sub-space
%   Detailed explanation goes here

% Step 1: Organize features
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

global samples_data_subjects_genuine;
global samples_data_subjects_fake;

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
%{
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


% Conduct Principle Component Analysis given samples from all subjects

% Assemble all samples
samples_data_allsubs = [];
for s = 1:SUB_NUM
    samples_data_onesub = [samples_data_subjects_genuine{s}; samples_data_subjects_fake{s}];
    samples_data_allsubs = [samples_data_allsubs; samples_data_onesub];
end

%}
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
%{
VirtualFeatures = cell(2, length(Feat_Type));

for b = 1:2
    for k = 1:length(Feat_Type)
        VirtualFeatures{b, k} = load(strcat('..\Domain_Adaptation\SynthFeatures_AllTSubs_', BTypes_fake{b}, '_', name{Feat_Type(k)}, '_', method));
    end
end

vsamples_data_subjects_fake = cell(SUB_NUM, 1);

for i = 1:SUB_NUM_T
    subid = clientID_target(i);
    s_rank = find(subid == clientID);
    s = s_rank(1);
    
    samples_data_q1 = [];
    for k = 1:length(Feat_Type)
        data = VirtualFeatures{1, k}.SynthFeatures_AllTSubs{i};
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
        data = VirtualFeatures{2, k}.SynthFeatures_AllTSubs{i};
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
    vsamples_data_subjects_fake{s} = [samples_data_q1; samples_data_q2]; 
end
%}

% close all;
% figure, hold on;

% I = eye(sum(dims(Feat_Type)), sum(dims(Feat_Type)));

cdists = zeros(SUB_NUM_T, 1);
stddevs = zeros(SUB_NUM_T, 1);
covs = zeros(SUB_NUM_T, 1);

load colors;
figure, hold on;
% Draw real samples in the 3-D sub-space
for i = 1:SUB_NUM
    % subid = clientID(i);
    % s_rank = find(subid == clientID);
    s = i;
    
    % samples_data_gen = samples_data_subjects_genuine{s};
    % samples_data_fake = samples_data_subjects_fake{s};
    % vsamples_data_fake = vsamples_data_subjects_fake{s};
    
    %{
    cdists(i) = sqrt(sum((mean(samples_data_fake, 1)-mean(vsamples_data_fake, 1)).^2));
    
    covmat = cov(vsamples_data_fake);
    covs(i) = sum((covmat(I == 1)));
    stddevs(i) = sum(sqrt(covmat(I == 1)));
    %}
    
    % Project the smaples into 3-D sub-space;
    
    % proj_samples_data_gen = samples_data_gen*ProjMat(:, 1:3);
    % proj_samples_data_fake = samples_data_fake*ProjMat(:, 1:3);
    % proj_vsamples_data_fake = vsamples_data_fake(1:end, :)*ProjMat(:, 1:3);
    
    % plot the samples in sub-space
    % plot the real genuine and fake samples from one subject
    figure, hold on;
    % display genuine samples with two different backgrounds
    legend_str = {};
    for b = 1:2
        ind = (strcmp(samples_btype_subjects_genuine{s}, BTypes_genuine{b}));
        legend_str = [legend_str, BTypes_genuine{b}];
        samples_gen_data_sub = samples_data_subjects_genuine{s}(ind, :);
        proj_samples_data_gen = samples_gen_data_sub*ProjMat(:, 1:3);
        plot3(proj_samples_data_gen(:, 1), proj_samples_data_gen(:, 2), proj_samples_data_gen(:, 3), 'Color', colors(b, :), 'LineStyle', '.');
    end
    
    % display genuine samples with various capturing conditions
    for b = 1:2
        % for m = 1:2
            for a = 1:3
                % for f = 1:2
                    ind = (strcmp(samples_btype_subjects_fake{s}, BTypes_fake{b}) & strcmp(samples_atype_subjects_fake{s}, ATypes_fake{a}));
                    legend_str = [legend_str, strcat(BTypes_fake{b}, {', '}, ATypes_fake{a})];
                    samples_fake_data_sub = samples_data_subjects_fake{s}(ind, :);
                    proj_samples_data_fake = samples_fake_data_sub*ProjMat(:, 1:3);
                    plot3(proj_samples_data_fake(:, 1), proj_samples_data_fake(:, 2), proj_samples_data_fake(:, 3), 'Color', colors(2 + 3 * (b-1) + a, :), 'LineStyle', '.');   
                % end
            end
        % end
    end
    legend(legend_str);
    % plot3(proj_samples_data_gen(:, 1), proj_samples_data_gen(:, 2), proj_samples_data_gen(:, 3), '.r');
    % plot3(proj_samples_data_fake(:, 1), proj_samples_data_fake(:, 2), proj_samples_data_fake(:, 3), '.g');   
%     
    % plot the virtual fake samples from one subject
    % plot3(proj_vsamples_data_fake(:, 1), proj_vsamples_data_fake(:, 2), proj_vsamples_data_fake(:, 3), '.b');       
    % close;
end

stat.cdists = cdists;
stat.stddevs = stddevs;
stat.covs = covs;

save(strcat('Stat_', name{Feat_Type}, '_', method), 'stat', '-v7.3');

end