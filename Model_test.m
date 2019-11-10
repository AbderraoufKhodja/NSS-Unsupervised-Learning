function output = Model_test(input, modelname, patchsize, one_im_dataset)
%%%-------------------------------------------------------------------------
%%% Model and database
%%%-------------------------------------------------------------------------
opts.imdbDir                                   = 'data/Set12';
opts.imdbPath                                  = fullfile(opts.imdbDir);
[imdb.inputs, ~, imdb.sorted_subs_dist_blocks] = Block_matching(input, patchsize, 27) ;
imdb.inputs                                    = reshape(imdb.inputs, patchsize, patchsize,1,[]);
opts.set                                       = 1:size(imdb.inputs,4);

%%%-------------------------------------------------------------------------
%%%  setting for simplenn
%%%-------------------------------------------------------------------------

opts.conserveMemory       = true;
opts.mode                 = 'test';
opts.cudnn                = true ;
opts.numSubBatches        = 1;
opts.Shuffle              = 0;

%%%-------------------------------------------------------------------------
%%%  setting for model
%%%-------------------------------------------------------------------------

opts.batchSize    = 2048*2;
opts.gpus         = [1];
opts.modelName    = modelname;
if one_im_dataset
    opts.expDir       = fullfile('data\trained_Nets\model_one_im',opts.modelName) ;
else
    opts.expDir       = fullfile('data\trained_Nets',opts.modelName) ;
end
opts.numberImdb   = 1;
opts.imdbDir      = opts.expDir;

%%%-------------------------------------------------------------------------
%%%  Initialization
%%%-------------------------------------------------------------------------
modelPath1 = @(ep) fullfile(opts.expDir, sprintf([opts.modelName,'-basic-epoch-%d.mat'], ep));
modelPath2 = @(ep) fullfile(opts.expDir, sprintf([opts.modelName,'-correction-epoch-%d.mat'], ep));

load(modelPath1(50), 'net1');
net1 = vl_simplenn_mergebnorm(net1);
net1 = vl_simplenn_tidy(net1) ;
load(modelPath2(100), 'net2');
net2 = vl_simplenn_mergebnorm(net2);
net2 = vl_simplenn_tidy(net2) ;

%%%-------------------------------------------------------------------------
%%%  Test
%%%-------------------------------------------------------------------------

if numel(opts.gpus) == 1
    net1 = vl_simplenn_move(net1, 'gpu') ;
    net2 = vl_simplenn_move(net2, 'gpu') ;
end

Patches = process_patches(net1, net2, imdb, opts);
Patches = gather(Patches);
output = col_to_im(Patches,[size(Patches,2) size(Patches,2)], size(input));
figure,imshow(output)

%%%-------------------------------------------------------------------------
    function  all_Patches = process_patches(net1, net2, imdb, opts)
%%%-------------------------------------------------------------------------
        subset = opts.set ;
        if numel(opts.gpus) == 1
            imdb.inputs = gpuArray(imdb.inputs);
        end
        for t=1:opts.batchSize:numel(subset)
            
            % get this image batch
            batchStart = t;
            batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
            batch = subset(batchStart : opts.numSubBatches : batchEnd) ;
            if numel(batch) == 0, continue ; end
            
            inputs = getBatch(imdb, batch) ;
            
            res1 = vl_simplenn(net1,inputs,[],[], 'mode', 'test', ...
                'conserveMemory', opts.conserveMemory, 'cudnn', opts.cudnn) ;
            
            res1 = vl_nnconcat({sum(res1(end).x,3)/8,res1(end).x},3);            
            res2 = vl_simplenn(net2, res1, [], [], 'mode', 'test', ...
                'conserveMemory', opts.conserveMemory, 'cudnn', opts.cudnn) ;
            
            if t~=1
                all_Patches = vl_nnconcat({all_Patches,res2(end).x},4);
            else
                all_Patches = res2(end).x;
            end
            disp(batch(end)*100/numel(subset))
        end
    end

%%%-------------------------------------------------------------------------
    function inputs = getBatch(imdb, batch)
        inputs = reshape(imdb.inputs(:,:,imdb.sorted_subs_dist_blocks(1:8,batch)), patchsize, patchsize, 8, []);
    end

    function [blocks, subs_dist_blocks, sorted_subs_dist_blocks] = Block_matching(image, patchsize , window )
        disp('Block matching...')
        usegpu = true;
        if usegpu == 1
            image = gpuArray(image);
        end
        image                    = im2single(image);
        blocks                   = im2col(image,[patchsize patchsize],'sliding');
        indices                  = reshape(1:size(blocks,2),size(image) - patchsize+1);
        window_indices           = im2col(padarray(indices,[(window-1)/2 (window-1)/2],'both'),[window window]);
        vals_dist_blocks         = gpuArray(zeros(size(window_indices),'single'));
        subs_dist_blocks         = zeros(size(window_indices));
        sorted_subs_dist_blocks  = zeros(9,size(window_indices,2));
        
        
        for i = 1:size(window_indices,1)
            index                 = window_indices(i,:);
            [~,col]               = find(index == 0);
            index(index == 0)     = col;
            inputs                = blocks(:,index);
            subs_dist_blocks(i,:) = index;
            vals_dist_blocks(i,:) = sum(bsxfun(@minus,inputs,blocks).^2)/(patchsize^2) ;
        end
        
        [val,ind] = sort(gather(vals_dist_blocks));
        for j=1:size(vals_dist_blocks,2)
            index = ind(find(val(:,j),8,'first'),j);
            sorted_subs_dist_blocks(:,j) = [j;window_indices(index,j)];
        end
    end
end
