# Person-Specific-Face-Anti-Spoofing
## Introduction
This project provides the codes used in the TIFS paper "[person Specific Face Anti-Spoofing with Subject Domain Adaptation](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=7041231&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D7041231)".
Different from previous works, we proposed to implement face anti-spoofing in person-specific manner, which means each subject
is equipped with a unique classifier for distinguishing genuine and fake face images. Moreover, we considered a challenging case that 
some subjects' fake face images are not available. Toward such a problem, we proposed a subject domain adaptation algorithm to 
generate virtual samples for training. The experimental results proves the superiority of person-specific face anti-spoofing method compared
with generic face anti-spoofing methods. Also, it is shown the proposed subject domain adaptation algorithm can generate virtual samples to 
assist the training of person-specific classifiers.
## Experiments
### Preparation
* libsvm: this open library can be found [here](http://www.csie.ntu.edu.tw/~cjlin/libsvm/).
* features: the off-the-shelf features extracted for this paper can be downloaded from my Baidu Driver [here](http://pan.baidu.com/s/1hqKmEpe) or Google Driver [here](https://drive.google.com/folderview?id=0B749j8XpVZQ-VFE0OG1hZFpFZXc&usp=sharing)

### Run the code

Use Replay-Attack dataset as the example. Go to REPLAY-ATTACK folder, and then:

1. Load Feats and Labels.

   We need load all features and labels for Replay-Attack dataset. Run
   ```bash
   load('Feats_Replay-Attack_tr_fall_soriginal.mat')  % load train set features
   load('Labels_Replay-Attack_tr_fall_soriginal.mat') % load train set labels
   load('Feats_Replay-Attack_dev_fall_soriginal.mat') % load development set features
   load('Labels_Replay-Attack_dev_fall_soriginal.mat') % load development set labels
   load('Feats_Replay-Attack_te_fall_soriginal.mat') % load test set features
   load('Labels_Replay-Attack_te_fall_soriginal.mat') % load test set labels
   load('Feats_Replay-Attack_en_fall_soriginal.mat') % load enrollment set features
   load('Labels_Replay-Attack_en_fall_soriginal.mat') % load enrollment set labels
   ```

2. Estimate Transformation.

    Go to folder Estimate_Transformationn_for_Expe, fund runall.m:
    ```bash
    Feat_Types = [1 3];
    BLabels = {'ContN' 'AdvN'};

    for t = 1:length(Feat_Types)
        for b = 1:length(BLabels)
            EstimateTransformation_Enroll_Iterative(Feats_enroll_SL, Labels_train_SL, Labels_devel_SL, Labels_test_SL, Labels_enroll_SL, BLabels{b}, Feat_Types(t), 'CS');
            EstimateTransformation_Enroll_Iterative(Feats_enroll_SL, Labels_train_SL, Labels_devel_SL, Labels_test_SL, Labels_enroll_SL, BLabels{b}, Feat_Types(t), 'OLS');
            EstimateTransformation_Enroll_Iterative(Feats_enroll_SL, Labels_train_SL, Labels_devel_SL, Labels_test_SL, Labels_enroll_SL, BLabels{b}, Feat_Types(t), 'PLS');
            EstimateTransformation_Enroll_PCA(Feats_enroll_SL, Labels_train_SL, Labels_devel_SL, Labels_test_SL, Labels_enroll_SL, Feat_Types(t), BLabels{b}, 'PCA');  
            % EstimateTransformation_Enroll_PCA_withSubNum(Feats_enroll_SL, Labels_train_SL, Labels_devel_SL, Labels_test_SL, Labels_enroll_SL, Feat_Types(t), BLabels{b}, 'PCA');
        end
    end	
	```

	In this file, we provided all methods to estimate the transformations, including CS, OLS, PLS as stated in the paper. We also provided a PCA-based transformation estimation. To get the transformation, comment the line as you need. Once finished, you will obtain a mat file with prefix "transform_".
	
3. Domain Adaptation.
    After get the estimated transformation, we can perform domain adaptation between subjects. Go to Domain_Adaptation folder, and find runall.m:
	```bash
	Feat_Types = [1 3];
    Methods = {'CS' 'OLS' 'PLS' 'PCA'};

    for t = 1:2
        for m = 1:4 % length(Methods)
            TargetDA_AllQualities(Feats_train_SL, Labels_train_SL, Feats_devel_SL, Labels_devel_SL, Feats_test_SL, Labels_test_SL, Feats_enroll_SL, Labels_enroll_SL, Feat_Types(t), Methods{m})        
        end
    end
	```
    Modify the variable *Methods* accordingly to do domain adaptation for different methods. Afterward, you will find mat files with prefix "SynthFeatures_".
	
4. Train Model.

    Congratulate to see you here. You are almost set! Go inside Train_Models folder. You can train generic face anti-spoofing model using runall_gen.m, while person-specific face anti-spoofing model using runall_ps.m. Meanwhile, you can also use runall.m to train models with the number of subjects used for training.
	
5. Test Model
	
	Finally, you may want to evaluate the performance. Go to Test_Models folder, find runall_ps.m and runall_gen.m and run them.
 
### Citation
If you find our code is useful in your researches, please consider citing:

    @article{yang2015person,
      title={Person-specific face antispoofing with subject domain adaptation},
      author={Yang, Jianwei and Lei, Zhen and Yi, Dong and Li, Stan Z},
      journal={IEEE Transactions on Information Forensics and Security},
      volume={10},
      number={4},
      pages={797--809},
      year={2015},
      publisher={IEEE}
    }
