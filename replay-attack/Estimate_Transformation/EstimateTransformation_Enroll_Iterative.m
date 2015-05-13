function EstimateTransformation_Enroll_Iterative(Feats_enroll_SL, Labels_train_SL, Labels_devel_SL, Labels_test_SL, Labels_enroll_SL, BLabel, Feat_Type, method)
%ESTIMATETRANSFORMATION_ITERATIVE Summary of this function goes here
%   Function: Estimate transformation from one subject domain to another one iteratively
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


% Step 1: Organize genuine training samples for all 20+30 subjects (first 1/3 part of all genuien samples)
name = {'MsLBP' 'LBP' 'HOG' 'LPQ'};
dims = [833 361 378 256];

SubIDs_train  = Labels_train_SL.SubID_train;
%{
PNLabels_train = Labels_train_SL.PNLabels_train;
BLabels_train = Labels_train_SL.BLabels_train;
MLabels_train = Labels_train_SL.MLabels_train;
ALabels_train = Labels_train_SL.ALabels_train;
FLabels_train = Labels_train_SL.FLabels_train;
Feats_train   = Feats_train_SL.Feats_train;
%}
SubIDs_devel   = Labels_devel_SL.SubID_devel;
%{
PNLabels_devel  = Labels_devel_SL.PNLabels_devel;
BLabels_devel  = Labels_devel_SL.BLabels_devel;
MLabels_devel  = Labels_devel_SL.MLabels_devel;
ALabels_devel = Labels_devel_SL.ALabels_devel;
FLabels_devel = Labels_devel_SL.FLabels_devel;
Feats_devel    = Feats_devel_SL.Feats_devel;
%}

SubIDs_test   = Labels_test_SL.SubID_test;
%{
PNLabels_test  = Labels_test_SL.PNLabels_test;
BLabels_test  = Labels_test_SL.BLabels_test;
MLabels_test  = Labels_test_SL.MLabels_test;
ALabels_test = Labels_test_SL.ALabels_test;
FLabels_test = Labels_test_SL.FLabels_test;
Feats_test    = Feats_test_SL.Feats_test;
%}

SubIDs   = Labels_enroll_SL.SubID_enroll;
PNLabels  = Labels_enroll_SL.PNLabels_enroll;
BLabels  = Labels_enroll_SL.BLabels_enroll;
MLabels  = Labels_enroll_SL.MLabels_enroll;
ALabels = Labels_enroll_SL.ALabels_enroll;
FLabels = Labels_enroll_SL.FLabels_enroll;
Feats   = Feats_enroll_SL.Feats_enroll;

clientID_source  = unique(SubIDs_train);
clientID_target = unique([SubIDs_devel, SubIDs_test]);
clientID = unique(SubIDs);
SUB_NUM = length(clientID);

samples_label_subject = zeros(length(SubIDs), 1);
samples_label_subjects = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    samples_label_subjects{s} = uint8(samples_label_subject);
end

for i = 1:length(SubIDs)
    s_rank = find(SubIDs(i)==clientID);
    s = s_rank(1);
    if strcmp(BLabels(i), BLabel) % && strcmp(ALabels(i), 'IpN') && strcmp(MLabels(i), 'FixN') && strcmp(FLabels(i), 'PhN')
        samples_label_subjects{s}(i) = 1;
    end
end

samples_data_subjects = cell(1, SUB_NUM);

for s = 1:SUB_NUM
    samples_data_subjects{s} = zeros(sum(samples_label_subjects{s}), dims(Feat_Type));
end

sample_sub_id = ones(1, SUB_NUM);

for i = 1:length(SubIDs)
    s_rank = find(SubIDs(i)==clientID);
    s = s_rank(1);
    if strcmp(BLabels(i), BLabel) % && strcmp(ALabels(i), 'IpN')  && strcmp(MLabels(i), 'FixN') && strcmp(FLabels(i), 'PhN')
        if Feat_Type == 1
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats{i}.MsLBP{1};
        elseif Feat_Type == 2
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats{i}.LBP{1};
        elseif Feat_Type == 3
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats{i}.HOG{1};
        elseif Feat_Type == 4
            samples_data_subjects{s}(sample_sub_id(s), :) = Feats{i}.LPQ{1};
        end
        sample_sub_id(s) = sample_sub_id(s) + 1;
    end
end

clear Feats;

% Use only the first 1/3 part of samples for estimating the transformation
%{
for s = 1:SUB_NUM
    samples_data_subjects{s} = samples_data_subjects{s}(1:3:end, :);
end
%}

% Step 2: Esitmate the transformations (Algorithm 1 in the paper)
H = cell(SUB_NUM, SUB_NUM);   % transformation from one subject to another
T = cell(SUB_NUM, SUB_NUM);   % Translation from one subject to another


% Step 2-1: Initialize transforamtion matrices
for m = 1:SUB_NUM
    for n = 1:SUB_NUM
        H{m, n} = 1;
        T{m, n} = 0;
    end
end

% Step 2-2: Iteration
for m = 1:SUB_NUM
    if sum(clientID(m) == clientID_source) == 0
        continue;
    end
    for n = 1:SUB_NUM
        
        if m == n || sum(clientID(n) == clientID_target) == 0
            continue;
        end
        disp(strcat('Adapt domain ', num2str(m), 'to domain ', num2str(n)));
        samples_s = samples_data_subjects{m};
        samples_t = samples_data_subjects{n};
        
        % L2 Normalization
        samples_s = bsxfun(@rdivide, samples_s, sqrt(sum(samples_s.^2, 2)));
        samples_t = bsxfun(@rdivide, samples_t, sqrt(sum(samples_t.^2, 2)));
        
        % If use OLS or PLS for estiamtion, then centralize the features
        
        if ~strcmp(method, 'CS')
          samples_s_c = bsxfun(@minus, samples_s, mean(samples_s, 1));
          samples_t_c = bsxfun(@minus, samples_t, mean(samples_t, 1));
        end
        
        % Initialize local variables
        Iter_time = 0;
        H_mn      = 1;
        T_mn      = zeros(1, dims(Feat_Type));
        L_p       = 10;
        
        while (1)
            % Given samples from two subject, estimate {H, T}

            if ~strcmp(method, 'CS')
                % Step 2-2-1: Using SVD update correspoindence
                % get the affinity matrix of two collection of samples
                AffinityMat = getAffinityMatrix(samples_s, samples_t);
                % get the correspondence accroding to affinity matrix
                pairs_matched = getCorrespondence(AffinityMat);
                if size(pairs_matched, 1) > 50
                    samples_s_matched = samples_s_c(pairs_matched(:, 1), :);
                    samples_t_matched = samples_t_c(pairs_matched(:, 2), :);
                else
                    num_min = min(size(AffinityMat));
                    samples_s_matched = samples_s_c(1:num_min, :);
                    samples_t_matched = samples_t_c(1:num_min, :);
                end
            else
                num_min = min(size(samples_s, 1), size(samples_t, 1));
                samples_s_matched = samples_s(1:num_min, :);
                samples_t_matched = samples_t(1:num_min, :);
            end

            %{
            if size(samples_s, 1) >= size(samples_t, 1)
                C_unique = unique(C);
                samples_s_matched = samples_s(C_unique, :);
                samples_t_matched = zeros(length(C_unique), size(samples_t, 2));
                for k = 1:length(C_unique)
                    idx = find(C == C_unique(k));
                    samples_t_matched(k, :) = samples_t(idx(1), :);
                end
                % samples_t_matched = samples_t;
            else
                R_unique = unique(R);
                samples_t_matched = samples_t(R_unique, :);
                samples_s_matched = zeros(length(R_unique), size(samples_s, 2));
                for k = 1:length(R_unique)
                    idx = find(R == R_unique(k));
                    samples_s_matched(k, :) = samples_s(idx(1), :);
                end
            end
            %}
            % Compute the loss 
            % Delta = bsxfun(@minus, samples_t_matched - H_mn*samples_s_matched, T_mn);
            % L_p = sqrt(trace(Delta*Delta'))/size(samples_s_matched, 1);
            
            % Step 2-2-2: Using 'method' to estimate transfomration
            if strcmp(method, 'CS')
                T_mn = mean(samples_t_matched, 1) - mean(samples_s_matched, 1);
                H_mn = 1;
                        
            elseif strcmp(method, 'OLS')
                X = [samples_s_matched'; ones(1, size(samples_s_matched, 1))];
                G_HT = samples_t_matched'*X'*inv(X*X' + 1e-7*eye(dims(Feat_Type)+1));
                H_mn = G_HT(:, 1: size(samples_s_matched, 2));
                T_mn = G_HT(:, end);
                
            elseif strcmp(method, 'PLS')
                X = samples_s_matched(1:end, :);
                Y = samples_t_matched(1:end, :);
                [~, ~, ~, ~, G_HT, ~, MSE] = plsregress(X, Y, min(50, floor(size(X, 1)/3)-1));  %dims_PLS(t)
                H_mn = G_HT(2:end, :)';
                T_mn = G_HT(1, :);
            end
            
            % Compute the loss after estimating new transformation
            Delta = bsxfun(@minus, samples_t_matched' - H_mn*samples_s_matched', T_mn(:));
            L_c = sqrt(trace(Delta'*Delta))/size(samples_s_matched, 1);
            
            if L_c < L_p
                % Update transforamtion and translation
                H{m, n} = H_mn*H{m, n};
                T{m, n} = bsxfun(@plus, H_mn*T{m, n}, T_mn(:));
                
                % Update samples in source domain
                samples_s = (bsxfun(@plus, H{m, n}*samples_s', T{m, n}(:, 1)))';
                if ~strcmp(method, 'CS')
                    samples_s_c = (bsxfun(@plus, H{m, n}*samples_s_c', T{m, n}(:, 1)))';
                end
            else
                break;
            end

            Iter_time = Iter_time + 1;
            % Suspend iteration if some conditions are satisfied
            if abs(L_c - L_p) < 1e-5 || Iter_time > 0
                break;
            end
            L_p = L_c;
        end
    end
end

% Step 3: Save the estimated transformations
Transform.H = H;
Transform.T = T;
save(strcat('Transform_Enroll_', BLabel, '_', name{Feat_Type}, '_', method, '.mat'), 'Transform', '-v7.3');

end

function affinitymat2d = getAffinityMatrix(samples_s, samples_t)

dist2d = pdist2(samples_s, samples_t);

affinitymat2d = exp(-dist2d);

end

function pairs_matched = getCorrespondence(affinitymat)

% Step 1-1: SVD for affinity matrix
[U,S,V] = svd(affinitymat);

% Step 1-2: find matchings for genuine samples of reference subjects
E = S;
E(logical(eye(min(size(S))))) = 1;
P = U*E*V';

% Step 1-3: convert P to Q

[~, C] = max(P);
[~, R] = max(P');

pairs_matched = [];

num_matched = 0;
for i = 1:length(C)
    if R(C(i)) == i
        pairs_matched = [pairs_matched; [C(i), i]];
        num_matched = num_matched + 1;
    end
end

end