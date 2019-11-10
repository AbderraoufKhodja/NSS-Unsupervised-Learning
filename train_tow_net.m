function [net1, net2, state1, state2] = DnCNN_train_two_net(net1, net2, imdb, varargin)

%    The function automatically restarts after each training epoch by
%    checkpointing.
%
%    The function supports training on CPU or on one or more GPUs
%    (specify the list of GPU IDs in the `gpus` option).

% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

%%%-------------------------------------------------------------------------
%%% solvers: SGD(default) and Adam with(default)/without gradientClipping
%%%-------------------------------------------------------------------------

%%% solver: Adam
%%% opts.solver = 'Adam';
opts.beta1   = 0.9;
opts.beta2   = 0.999;
opts.alpha   = 0.01;
opts.epsilon = 1e-8;


%%% solver: SGD
opts.solver       = 'SGD';
opts.learningRate = logspace(-1,-4,50);
opts.weightDecay  = 0.0001;
opts.momentum     = 0.9 ;

%%% GradientClipping
opts.gradientClipping = false;
opts.theta            = 0.05;

%%% specific parameter for Bnorm
opts.bnormLearningRate = 0;

%%%-------------------------------------------------------------------------
%%%  setting for simplenn
%%%-------------------------------------------------------------------------

opts.conserveMemory       = true;
opts.mode                 = 'normal';
opts.cudnn                = true ;
opts.backPropDepth        = +inf ;
opts.skipForward          = false;
opts.numSubBatches        = 1;
opts.Dignostic            = 0;
opts.Shuffle              = 1;
opts.NumBatches           = 1600*4;
%%%-------------------------------------------------------------------------
%%%  setting for model
%%%-------------------------------------------------------------------------

opts.batchSize   = 128 ;
opts.gpus        = 1;
opts.numEpochs   = 300 ;
opts.modelName   = 'model';
opts.expDir      = fullfile('data',opts.modelName) ;
opts.numberImdb  = 1;
opts.imdbDir     = opts.expDir;

%%%-------------------------------------------------------------------------
%%%  update settings
%%%-------------------------------------------------------------------------

opts = vl_argparse(opts, varargin);
opts.numEpochs = numel(opts.learningRate);

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end

%%%-------------------------------------------------------------------------
%%%  Initialization
%%%-------------------------------------------------------------------------

net1 = vl_simplenn_tidy(net1);    %%% fill in some eventually missing values
net1.layers{end-1}.precious = 1;
vl_simplenn_display(net1, 'batchSize', opts.batchSize) ;
net2 = vl_simplenn_tidy(net2);    %%% fill in some eventually missing values
net2.layers{end-1}.precious = 1;
vl_simplenn_display(net2, 'batchSize', opts.batchSize) ;
state1.getBatch = getBatch ;
%%%-------------------------------------------------------------------------
%%%  Train and Test
%%%-------------------------------------------------------------------------
modelPath1 = @(ep) fullfile(opts.expDir, sprintf([opts.modelName,'basic-epoch-%d.mat'], ep));
modelPath2 = @(ep) fullfile(opts.expDir, sprintf([opts.modelName,'correction-epoch-%d.mat'], ep));

start = findLastCheckpoint(opts.expDir,opts.modelName) ;
if start >= 1
    fprintf('%s: resuming by loading epoch %d', mfilename, start) ;
    if start+1<opts.numEpochs/2
        load(modelPath1(start), 'net1','state1') ;
        net1 = vl_simplenn_tidy(net1) ;
    end
    if start+1 == opts.numEpochs/2+1
        load(modelPath1(opts.numEpochs/2), 'net1','state1') ;
        net1 = vl_simplenn_tidy(net1) ;
    end
    if start+1>opts.numEpochs/2+1
        load(modelPath1(opts.numEpochs/2), 'net1','state1') ;
        net1 = vl_simplenn_tidy(net1) ;
        load(modelPath2(start), 'net2','state2') ;
        net2 = vl_simplenn_tidy(net2) ;
    end
end

%%% Eval set
rng(0);
[eval_set,~,eval_labels] = state1.getBatch(imdb, randi(numel(find(imdb.set == 1)),1,1000)) ;
rng('shuffle');

if numel(opts.gpus) == 1
    eval_set = gpuArray(eval_set);
    eval_labels = gpuArray(eval_labels);
end
figure(1)
for epoch = start+1 : opts.numEpochs
    opts.train = find(imdb.set == 1);
    
    %%% Train for one epoch.
    state1.epoch = epoch ;
    state2.epoch = state1.epoch ;
    state1.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate)));
    state2.learningRate = state1.learningRate;
    opts.thetaCurrent = opts.theta(min(epoch, numel(opts.theta)));
    
    if numel(opts.gpus) == 1
        net1 = vl_simplenn_move(net1, 'gpu') ;
        net2 = vl_simplenn_move(net2, 'gpu') ;  
    end
    rng('shuffle');
    if opts.Shuffle == 1
        state1.train = opts.train(randperm(numel(opts.train))) ; %%% shuffle
        state2.train = state1.train;
    else
        state1.train = opts.train ; %%% no shuffle
        state2.train = state1.train;
    end
    if opts.NumBatches~=0
        state1.train = state1.train(1:opts.batchSize*opts.NumBatches);
        state2.train = state1.train;
    end
    [net1, net2, state1, state2] = process_epoch(net1, net2, state1, state2, imdb, opts, 'train');
    
    %%% Evalution Set
    %%% CL_Net
    if  state1.epoch<opts.numEpochs/2+1
        
        net1.layers{end}.class  = eval_labels(:,:,1:8,:) ;
        res1 = vl_simplenn(net1, eval_set(:,:,1:8,:), [], [], ...
            'mode', 'test', ...
            'conserveMemory', opts.conserveMemory, ...
            'cudnn', opts.cudnn) ;
        state1.eval_loss_track_1(epoch) = gather(res1(end).x)/size(res1(end-1).x,3);
    end
    
    %%% Ag_Net
    if state2.epoch>opts.numEpochs/2
        net1.layers{end}.class = eval_labels(:,:,1,:) ;
        eval_set_est = vl_simplenn(net1,eval_set(:,:,1:8,:),[],[],'conserveMemory',true,'mode','test');
        state1.eval_loss_track_1(epoch) = gather(eval_set_est(end).x)/size(eval_set_est(end-1).x,3);
        eval_set_est = vl_nnconcat({sum(eval_set_est(end-1).x,3)/8,eval_set_est(end-1).x},3);
        
        net2.layers{end}.class = eval_labels(:,:,1,:) ;
        res2 = vl_simplenn(net2, eval_set_est(:,:,1:9,:), [], [], ...
            'mode', 'test', ...
            'conserveMemory', opts.conserveMemory, ...
            'cudnn', opts.cudnn) ;
        state2.eval_loss_track_2(epoch) = gather(res2(end).x);
    end
    
    net1.layers{end}.class =[];
    net2.layers{end}.class =[];
    net1 = vl_simplenn_move(net1, 'cpu');
    net2 = vl_simplenn_move(net2, 'cpu');
    
    %%% save current model
    if state1.epoch<opts.numEpochs/2+1
        hold on
        plot(mean(state1.real_loss_track_1,2))
        plot(state1.eval_loss_track_1)
        hold off
        save(modelPath1(epoch), 'net1','state1')
    end
    if state2.epoch>opts.numEpochs/2
        hold on
        plot(mean(state1.real_loss_track_1,2))
        plot(state1.eval_loss_track_1)
        plot(mean(state2.real_loss_track_2(opts.numEpochs/2+1:end,:),2))
        plot(state2.eval_loss_track_2(opts.numEpochs/2+1:end))
        hold off
        save(modelPath2(epoch), 'net2','state2')
    end
    
end


%%%-------------------------------------------------------------------------
function  [net1, net2, state1, state2] = process_epoch(net1, net2, state1, state2, imdb, opts, mode)
%%%-------------------------------------------------------------------------
if strcmp(mode,'train')
    
    switch opts.solver
        
        case 'SGD' %%% solver: SGD
            for i = 1:numel(net1.layers)
                if isfield(net1.layers{i}, 'weights')
                    for j = 1:numel(net1.layers{i}.weights)
                        state1.layers{i}.momentum{j} = 0;
                    end
                end
            end
            for i = 1:numel(net2.layers)
                if isfield(net2.layers{i}, 'weights')
                    for j = 1:numel(net2.layers{i}.weights)
                        state2.layers{i}.momentum{j} = 0;
                    end
                end
            end
            
        case 'Adam' %%% solver: Adam
            for i = 1:numel(net2.layers)
                if isfield(net2.layers{i}, 'weights')
                    for j = 1:numel(net2.layers{i}.weights)
                        state1.layers{i}.t{j} = 0;
                        state1.layers{i}.m{j} = 0;
                        state1.layers{i}.v{j} = 0;
                        
                        state2.layers{i}.t{j} = 0;
                        state2.layers{i}.m{j} = 0;
                        state2.layers{i}.v{j} = 0;
                    end
                end
            end
            
    end
    
end


subset = state1.(mode) ;
num = 0 ;
res1 = [];
res2 = [];

for t=1:opts.batchSize:numel(subset)
    
    for s=1:opts.numSubBatches
        if s>1
            disp('Sub_batches not supported');
            break;
        end
        % get this image batch
        batchStart = t + (s-1);
        batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : opts.numSubBatches : batchEnd) ;
        num = num + numel(batch) ;
        if numel(batch) == 0 || numel(batch)~=opts.batchSize/opts.numSubBatches , continue ; end
        
        [inputs,labels,orig_labels] = state1.getBatch(imdb, batch) ;
        
        if numel(opts.gpus) == 1
            inputs = gpuArray(inputs);
            labels = gpuArray(labels);
        end
        
        if strcmp(mode, 'train')
            dzdy1 = single(1);
            evalMode1 = 'normal';%%% forward and backward (Gradients)
        else
            dzdy1 = [] ;
            evalMode1 = 'test';  %%% forward only
        end
        
        if strcmp(mode, 'train')
            dzdy2 = single(1);
            evalMode2 = 'normal';%%% forward and backward (Gradients)
        else
            dzdy2 = [] ;
            evalMode2 = 'test';  %%% forward only
        end
        
        %%% CL_Net
        if  state1.epoch<opts.numEpochs/2+1
            
            net1.layers{end}.class  = labels ;
            permutation             = randperm(8)+1;
            res1 = vl_simplenn(net1, inputs(:,:,permutation,:), dzdy1, res1, ...
                'accumulate', s ~= 1, ...
                'mode', evalMode1, ...
                'conserveMemory', opts.conserveMemory, ...
                'backPropDepth', opts.backPropDepth, ...
                'cudnn', opts.cudnn) ;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Ag_Net
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if state2.epoch>opts.numEpochs/2
            net1.layers{end}.class = labels ;
            permutation            = randperm(9);
            res1 = vl_simplenn(net1,inputs(:,:,1:8,:),[],[],'conserveMemory',true,'mode','test');
            inputs = vl_nnconcat({sum(res1(end-1).x,3)/8,res1(end-1).x},3);
            inputs = inputs(:,:,permutation,:);
            net2.layers{end}.class = labels ;
            
            res2 = vl_simplenn(net2, inputs, dzdy2, res2, ...
                'accumulate', s ~= 1, ...
                'mode', evalMode2, ...
                'conserveMemory', opts.conserveMemory, ...
                'backPropDepth', opts.backPropDepth, ...
                'cudnn', opts.cudnn) ;
        end
        
    end
    
    %%% Update CL_Net
    if  state1.epoch<opts.numEpochs/2+1 && strcmp(mode, 'train')
        [state1, net1] = params_updates(state1, net1, res1, opts, opts.batchSize) ;
    end
    
    %%% Update Ag_Net
    if state2.epoch>opts.numEpochs/2 && strcmp(mode, 'train')
        [state2, net2] = params_updates(state2, net2, res2, opts, opts.batchSize) ;
    end
    
    %% Print real Loss
    if  state1.epoch<opts.numEpochs/2+1
        lossL2_1 = gather(res1(end).x)/size(res1(end-1).x,3) ;
        state1.loss_track_1(state1.epoch,fix((t-1)/opts.batchSize)+1) = lossL2_1;
        state1.real_loss_track_1(state1.epoch,fix((t-1)/opts.batchSize)+1) = ...
            gather(vl_nnloss(orig_labels(:,:,permutation,:),res1(end-1).x))/size(res1(end-1).x,3);
        fprintf('%s: epoch %02d dataset %02d: %3d/%3d:', mode, state1.epoch, mod(state1.epoch,opts.numberImdb), ...
            fix((t-1)/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
        fprintf('error: %f Groundtruth Error %f \n', lossL2_1, state1.real_loss_track_1(state1.epoch,fix((t-1)/opts.batchSize)+1)) ;
    end
    
    if state2.epoch>opts.numEpochs/2
        lossL2_2 = gather(res2(end).x) ;
        state2.loss_track_2(state2.epoch,fix((t-1)/opts.batchSize)+1) = lossL2_2;
        state2.real_loss_track_2(state2.epoch,fix((t-1)/opts.batchSize)+1) = ...
            gather(vl_nnloss(orig_labels(:,:,1,:),res2(end-1).x));
        fprintf('%s: epoch %02d dataset %02d: %3d/%3d:', mode, state2.epoch, mod(state2.epoch,opts.numberImdb), ...
            fix((t-1)/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
        fprintf('error: %f Groundtruth Error %f\n', lossL2_2, state2.real_loss_track_2(state2.epoch,fix((t-1)/opts.batchSize)+1)) ;
    end
    %% Diagnostic
    if opts.Dignostic == 1
        vl_simplenn_diagnose(net, res)
    end
    
end


%%%-------------------------------------------------------------------------
function [state, net] = params_updates(state, net, res, opts, batchSize)
%%%-------------------------------------------------------------------------

switch opts.solver
    
    case 'SGD' %%% solver: SGD
        
        for l=numel(net.layers):-1:1
            for j=1:numel(res(l).dzdw)
                if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
                    %%% special case for learning bnorm moments
                    thisLR = net.layers{l}.learningRate(j) - opts.bnormLearningRate;
                    net.layers{l}.weights{j} = vl_taccum(...
                        1 - thisLR, ...
                        net.layers{l}.weights{j}, ...
                        thisLR / batchSize, ...
                        res(l).dzdw{j}) ;
                    
                else
                    thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j);
                    thisLR = state.learningRate * net.layers{l}.learningRate(j);
                    
                    if opts.gradientClipping
                        theta = opts.thetaCurrent/thisLR;
                        state.layers{l}.momentum{j} = opts.momentum * state.layers{l}.momentum{j} ...
                            - thisDecay * net.layers{l}.weights{j} ...
                            - (1 / batchSize) * gradientClipping(res(l).dzdw{j},theta) ;
                        net.layers{l}.weights{j} = net.layers{l}.weights{j} + ...
                            thisLR * state.layers{l}.momentum{j} ;
                    else
                        state.layers{l}.momentum{j} = opts.momentum * state.layers{l}.momentum{j} ...
                            - thisDecay * net.layers{l}.weights{j} ...
                            - (1 / batchSize) * res(l).dzdw{j} ;
                        net.layers{l}.weights{j} = net.layers{l}.weights{j} + ...
                            thisLR * state.layers{l}.momentum{j} ;
                    end
                end
            end
        end
        
        
    case 'Adam'  %%% solver: Adam
        for l=numel(net.layers):-1:1
            for j=1:numel(res(l).dzdw)
                
                if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
                    
                    % special case for learning bnorm moments
                    thisLR = net.layers{l}.learningRate(j);
                    net.layers{l}.weights{j} = vl_taccum(...
                        1 - thisLR, ...
                        net.layers{l}.weights{j}, ...
                        thisLR / batchSize, ...
                        res(l).dzdw{j}) ;
                    
                else
                    
                    %  if   j == 1 && strcmp(net.layers{l}.type, 'bnorm')
                    %         c = net.layers{l}.weights{j};
                    %         net.layers{l}.weights{j} = clipping(c,mean(abs(c))/2);
                    %  end
                    
                    thisLR = state.learningRate * net.layers{l}.learningRate(j);
                    state.layers{l}.t{j} = state.layers{l}.t{j} + 1;
                    t = state.layers{l}.t{j};
                    alpha = thisLR;
                    lr = alpha * sqrt(1 - opts.beta2^t) / (1 - opts.beta1^t);
                    
                    state.layers{l}.m{j} = state.layers{l}.m{j} + (1 - opts.beta1) .* (res(l).dzdw{j} - state.layers{l}.m{j});
                    state.layers{l}.v{j} = state.layers{l}.v{j} + (1 - opts.beta2) .* (res(l).dzdw{j} .* res(l).dzdw{j} - state.layers{l}.v{j});
                    
                    % weight decay
                    net.layers{l}.weights{j} = net.layers{l}.weights{j} -  thisLR * opts.weightDecay * net.layers{l}.weightDecay(j) * net.layers{l}.weights{j};
                    
                    % update weights
                    net.layers{l}.weights{j} = net.layers{l}.weights{j} - lr * state.layers{l}.m{j} ./ (sqrt(state.layers{l}.v{j}) + opts.epsilon) ;
                    
                end
            end
        end
        
end


%%%-------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir,modelName)
%%%-------------------------------------------------------------------------
list = dir(fullfile(modelDir, [modelName,'basic-epoch-*.mat'])) ;
tokens = regexp({list.name}, [modelName,'basic-epoch-([\d]+).mat'], 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

%%%-------------------------------------------------------------------------
function A = gradientClipping(A, theta)
%%%-------------------------------------------------------------------------
A(A>theta)  = theta;
A(A<-theta) = -theta;

%%%-------------------------------------------------------------------------
function fn = getBatch
%%%-------------------------------------------------------------------------
fn = @(x,y) getSimpleNNBatch(x,y);


%%%-------------------------------------------------------------------------
function [inputs,ref_labels,orig_labels] = getSimpleNNBatch(imdb, batch)
%%%-------------------------------------------------------------------------
rng('shuffle');
mode = rand(8);
inputs = zeros(size(imdb.inputs_noisy,1),size(imdb.inputs_noisy,2),9,numel(batch),'single');
orig_labels = zeros(size(imdb.inputs_orig,1),size(imdb.inputs_orig,2),9,numel(batch),'single');
for i=1:9
    inputs(:,:,i,:) = imdb.inputs_noisy(:,:,:,imdb.sorted_subs_dist_blocks(i,batch));
end
inputs = data_augmentation(inputs, mode);

ref_labels = inputs(:,:,1,:);

%% For tracking the real loss
for i=1:9
    orig_labels(:,:,i,:) = imdb.inputs_orig(:,:,:,imdb.sorted_subs_dist_blocks(i,batch));
end
orig_labels = data_augmentation(orig_labels, mode);
