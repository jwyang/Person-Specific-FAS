function HTER = lfr_process(Perf, FRLabels, indice, clientID)

    thresholds = Perf.thresholds;
    SUB_NUM = length(Perf.thresholds);
    FANum = zeros(SUB_NUM, 1);
    FRNum = zeros(SUB_NUM, 1);
    
    nums_gen = zeros(SUB_NUM, 1);
    nums_fake = zeros(SUB_NUM, 1);
    
    HTERs = zeros(length(indice), 1);
    
    for x = 1:length(indice)
        % 
        s = indice(x);
        
        gen_num = length(Perf.Scores_gen{s, s});
        fake_num = length(Perf.Scores_fake{s, s});
        
        nums_gen(s) = gen_num;
        nums_fake(s) = fake_num;
        
        genLabels = FRLabels{1, s};
        fakeLabels = FRLabels{2, s};
                
        FANum(s) = sum(Perf.Scores_fake{s, s} >= (thresholds(s)) & fakeLabels == clientID(s)); % 
        
        % Add the number fake Samples which is incorrectly identified and recognized as genuine faces
        ind = find(fakeLabels ~= clientID(s));
        for i = 1:length(ind)
            idx = find(fakeLabels(ind(i)) == clientID);
            if isempty(idx)
                nums_fake(s) = nums_fake(s) - 1;
                continue;
            end
            FANum(s) = FANum(s) + (Perf.Scores_fake{s, idx(1)}(ind(i)) >= thresholds(idx(1)));
        end
        
        % FRNum(s) = sum(Perf.Scores_gen{s, s} < (thresholds(s)) | genLabels ~= s);    % 
        FRNum(s) = sum(Perf.Scores_gen{s, s} < (thresholds(s)) & genLabels == clientID(s));    % 
        ind = find(genLabels ~= clientID(s));
        for i = 1:length(ind)
            idx = find(genLabels(ind(i)) == clientID);
            if isempty(idx)
                nums_gen(s) = nums_gen(s) - 1;
                continue;
            end
            FRNum(s) = FRNum(s) + (Perf.Scores_gen{s, idx(1)}(ind(i)) < thresholds(idx(1)));
        end
        
        HTERs(x) = (FANum(s) / nums_fake(s) + FRNum(s) / nums_gen(s))/ 2;
    end
    

    FAR = sum(FANum)/sum(nums_fake);
    FRR = sum(FRNum)/sum(nums_gen);
    
    HTER = mean(HTERs); % (FAR + FRR)/2;
end