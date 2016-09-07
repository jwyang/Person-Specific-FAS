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

1. Load feats and labels.

   We need load all features for Replay-Attack dataset. Run
   ```bash
   >> load('Feats_Replay-Attack_tr_fall_soriginal.mat')  % load train set features
   >> load('Labels_Replay-Attack_tr_fall_soriginal.mat') % load train set labels
   >> load('Feats_Replay-Attack_dev_fall_soriginal.mat') % load development set features
   >> load('Labels_Replay-Attack_dev_fall_soriginal.mat') % load development set labels
   >> load('Feats_Replay-Attack_te_fall_soriginal.mat') % load test set features
   >> load('Labels_Replay-Attack_te_fall_soriginal.mat') % load test set labels
   >> load('Feats_Replay-Attack_en_fall_soriginal.mat') % load enrollment set features
   >> load('Labels_Replay-Attack_en_fall_soriginal.mat') % load enrollment set labels
   ```

2. Estimate Transformation.

    Go to folder Estimate_Transformationn_for_Expe, fun runall.m. Then run:
	
	
3. Domain Adaptation.


4. Train Model.

5. Test Model
	
 
## Keynotes
