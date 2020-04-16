
======================================================================

Similarity Noise Training for Image Denoising

Contact: KHODJA ABD ERRAOUF, Email:201710800010@zjnu.edu.cn

----------------------------------------------------------------------
Training and testing demo Similarity Noise Training for Image Denoising, 
All training have been condacted on Windows 10, with Matlab R2018b with
Matconvnet toolbox consider compiling Matconvnet in advance.


------ Contents -----------------------------------------------------

data

----Test (Set12 and Set68 )

----Train400 (400 training images default training patch size 20X20)

----utilities 

----model_noise_NSS_no_gt_training.m             (run this to generate training (clean) patches!)

model_noise_NSS_no_gt_training.m                 (run this to train the model) 

test_performance_one_image_NSS_no_gt_training.m  (test  one image model)

test_performance_Set12_NSS_no_gt_training.m      (test Set12 dataset model)

test_performance_Train400_NSS_no_gt_training.m   (test  Train400 model)

train_tow_net.m       (the main body of training code)

Model_test            (Helper function for performing patch matching and testing)

Ag_Net.m, CL_Net.m    (initializate the model)

README.tex

README.md

----------------------------------------------------------------------
