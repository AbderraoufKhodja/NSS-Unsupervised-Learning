rng('default')
addpath('utilities')
vl_setupnn
%%%-------------------------------------------------------------------------
%%% Configuration
%%%-------------------------------------------------------------------------
opts.noiseSigma       = 25;
opts.learningRate     = [logspace(-3,-6,50),logspace(-3,-8,50)];%%% Chooose learning rate default [logspace(-3,-6,50),logspace(-3,-8,50)]
opts.batchSize        = 64;
opts.gpus             = [1]; %%% this code can only support one GPU!
opts.numSubBatches    = 1;
opts.bnormLearningRate= 0;


%%% solver
opts.solver           = 'SGD';
opts.numberImdb       = 1;

%%%
opts.imdbDir          = ['data/one_image'
    % 'data/Set12'
    % 'data/Train400'
    ];
opts.modelName        = sprintf('model_noise_%i_one_image_7_NSS_no_gt_training',opts.noiseSigma); %%% model name your image on folder one_im for one image trainnig
%opts.modelName       = sprintf('model_noise_%i_Set12_NSS_no_gt_training',opts.noiseSigma); %%% model name
%opts.modelName       = sprintf('model_noise_%i_Train400_NSS_no_gt_training',opts.noiseSigma); %%% model name
opts.imdbPath         = fullfile(opts.imdbDir);
imdb                  = block_matching_patches(opts.imdbPath, opts.noiseSigma) ;
opts.gradientClipping = true; %%% set 'true' to prevent exploding gradients in the beginning.
opts.backPropDepth    = Inf;
opts.NumBatches       = 100; %%% default 400
%%%------------------------------------------------------------------------
%%%   Initialize model and load data
%%%------------------------------------------------------------------------
%%%  model
net1                        = feval('CL_Net');
net2                        = feval('Ag_Net');

%%%  load data
opts.expDir      = fullfile('data', opts.modelName);
%%%-------------------------------------------------------------------------
%%%   Train
%%%-------------------------------------------------------------------------

[net1, net2, info1, info2] = train_tow_net(net1, net2, imdb,  ...
    'expDir', opts.expDir, ...
    'learningRate',opts.learningRate, ...
    'bnormLearningRate',opts.bnormLearningRate, ...
    'numSubBatches',opts.numSubBatches, ...
    'numberImdb',opts.numberImdb, ...
    'backPropDepth',opts.backPropDepth, ...
    'imdbDir',opts.imdbDir, ...
    'solver',opts.solver, ...
    'gradientClipping',opts.gradientClipping, ...
    'batchSize', opts.batchSize, ...
    'modelname', opts.modelName, ...
    'NumBatches',opts.NumBatches, ...
    'gpus',opts.gpus) ;