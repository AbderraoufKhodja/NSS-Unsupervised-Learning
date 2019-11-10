function imdb = block_matching_patches(folder, noiseSigma)

addpath('utilities');
batchSize     = 64;        %%% batch size
dataName      = 'TrainingPatches';

patchsize     = 20;
stride        = 1;
step          = 0;
window        = 27;
blocksize     = 20;

column_wise   = 0;
count1        = 0;

ext               =  {'*.jpg','*.png','*.bmp','*.jpeg'};
filepaths           =  [];

for i = 1 : length(ext)
    filepaths = cat(1,filepaths, dir(fullfile(folder, ext{i})));
end

%% count1 the number of extracted patches
for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder,filepaths(i).name)); % uint8
    [hei,wid,~] = size(image);
    for x = 1+step : stride : (hei-patchsize+1)
        for y = 1+step :stride : (wid-patchsize+1)
            count1 = count1+1;
        end
    end
    if mod(i,100)==0
        disp([i,length(filepaths)]);
    end
end

numPatches = ceil(count1/batchSize)*batchSize;

% pause;
inputs_noisy            = zeros(patchsize,patchsize, 1, numPatches,'single'); % this is fast
inputs_orig             = zeros(patchsize,patchsize, 1, numPatches,'single'); % this is fast
sorted_subs_dist_blocks = zeros(9,numPatches);
tic;
start = 1;
for i = 1 : length(filepaths)
    rng(0)
    image_gt = imread(fullfile(folder,filepaths(i).name)); % uint8
    num_im_blocks = size(im2col(image_gt,[patchsize patchsize],'s'),2);
    if mod(i,100)==0
        disp([i,length(filepaths)]);
    end
    im_gt         = im2single(image_gt); % single
    im_noise      = im_gt + noiseSigma/255.*randn(size(im_gt)); % single
    start_end     = start:start+num_im_blocks-1;
    [~, ~, inputs_noisy(:,:,:,start_end),inputs_orig(:,:,:,start_end),sorted_subs_dist_blocks(:,start_end)] = Block_maching(im_noise, im_gt, patchsize , window, column_wise );
    sorted_subs_dist_blocks(:,start_end) =  sorted_subs_dist_blocks(:,start_end) + start-1;
        
start = start + num_im_blocks;
end
set    = uint8(ones(1,size(inputs_noisy,4)));
[~,col]= find(~sorted_subs_dist_blocks,1);
set(col:end)=0;

disp('-------Datasize-------')
disp([size(inputs_noisy,4),batchSize,size(inputs_noisy,4)/batchSize]);

imdb.inputs_orig = inputs_orig;
imdb.inputs_noisy = inputs_noisy;
imdb.sorted_subs_dist_blocks= sorted_subs_dist_blocks;
imdb.set = set;

toc;