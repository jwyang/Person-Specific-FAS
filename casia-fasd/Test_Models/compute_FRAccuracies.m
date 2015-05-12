function compute_FRAccuracies(Feat_Type)
%COMPUTE_FRACCURACIES Summary of this function goes here
%   Detailed explanation goes here

name = {'MsLBP' 'LBP' 'HOG' 'LPQ'};
modelname = '';
for k = 1:length(Feat_Type)
    modelname = strcat(modelname, '_', name{Feat_Type(k)});
end

% load(strcat('FRPerf_test', modelname, '.mat'));
load(strcat('FRPerf_test_withSynth', modelname, '_', 'CS', '.mat'));
load(strcat('GenericPerf_test',modelname));

samples_num = 0;
samples_correct_num = 0;
for s = 21:50
    ind = FRLabels{s}.inds;
    samples_num = samples_num + length(ind);
    samples_correct_num = samples_correct_num + sum(ind == s);
end

accuracy = samples_correct_num/samples_num;

disp(strcat('Face Recognition Accuracy: ', num2str(accuracy)));


end

