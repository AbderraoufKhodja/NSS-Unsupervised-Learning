%%% test the model performance
vl_setupnn
format compact

folderTest  = fullfile('data','Set12'); %%% test dataset
patchsize   = 7;  %%% Default patch size [20 20] A smaller patch size would
                  %%% reduce compututinal cost a tthe expence of denosiing performance
showResult  = 1;
useGPU      = 1;
noiseSigma  = 25;  %%% [15 25 50] image noise level

%%% read images
ext         =  {'*.jpg','*.png','*.bmp'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
end

%% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));

for i = 1:length(filePaths)
    
    %%% read images
    label = imread(fullfile(folderTest,filePaths(i).name));
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    label = im2double(label);
    [w,h,~]=size(label);
    if size(label,3)==3
        label = rgb2gray(label);
    end
    %%% Select the model
    modelname   = sprintf('model_noise_%i_Set12_NSS_no_gt_training', noiseSigma);
    one_im_dataset = 0;
    %%% Add noise
  
    randn('seed', 0);
    input     = single(label + noiseSigma/255*randn(size(label)));
    output    = Model_test(input, modelname, patchsize, one_im_dataset);
    
    %%% Saving the results
    % imwrite(gather(input), fullfile(folderEst, filePaths(i).name));
    [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label), im2uint8(output),0,0);
    PSNRs(i) = gather(PSNRCur);
    SSIMs(i) = gather(SSIMCur);
    
    %%% convert to CPU
    if useGPU
        output = gather(output);
        input  = gather(input);
    end
    
    if showResult
        imshow(cat(2,im2uint8(label),im2uint8(input),im2uint8(output)));
        title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        drawnow;
    end
    
end