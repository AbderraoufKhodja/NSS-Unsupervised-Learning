function [vals_dist_blocks, subs_dist_blocks, sqr_blocks1, sqr_blocks2, sortedsubs_dist_blocks] = Block_maching(image1, image2, patchsize , window, column_wise )
disp('Block matching...')
usegpu = true;
if usegpu == 1
    image1 = gpuArray(image1);
end
image1                   = im2single(image1);
blocks1                  = im2col(image1,[patchsize patchsize],'sliding');
blocks2                  = im2col(image2,[patchsize patchsize],'sliding');
sqr_blocks1              = zeros(patchsize,patchsize,size(blocks1,2));
sqr_blocks2              = zeros(patchsize,patchsize,size(blocks1,2));
indices                  = reshape(1:size(blocks1,2),size(image1) - patchsize+1);
window_indices           = im2col(padarray(indices,[(window-1)/2 (window-1)/2],'both'),[window window]);
vals_dist_blocks         = gpuArray(zeros(size(window_indices),'single'));
subs_dist_blocks         = zeros(size(window_indices));
sortedsubs_dist_blocks   = zeros(9,size(window_indices,2));


for i = 1:size(window_indices,1)
    index                 = window_indices(i,:);
    [~,col]               = find(index == 0);
    index(index == 0)     = col;
    inputs                = blocks1(:,index);
    subs_dist_blocks(i,:) = index;
    vals_dist_blocks(i,:) = sum(bsxfun(@minus,inputs,blocks1).^2)/(patchsize^2) ;
    
end

vals_dist_blocks = gather(vals_dist_blocks) ;

[val,ind]=sort(vals_dist_blocks);
blocks1 = gather(blocks1);
blocks2 = gather(blocks2);
if column_wise == ~1
    for j=1:size(vals_dist_blocks,2)
        index = ind(find(val(:,j),8,'first'),j);
        sortedsubs_dist_blocks(:,j) = [j;window_indices(index,j)];
        sqr_blocks1(:,:,j) = reshape(blocks1(:,j),[patchsize patchsize]);
        sqr_blocks2(:,:,j) = reshape(blocks2(:,j),[patchsize patchsize]);
    end
else
    for j=1:size(vals_dist_blocks,2)
        index = ind(find(val(:,j),8,'first'),j);
        sortedsubs_dist_blocks(:,j) = [j;window_indices(index,j)];
    end
    sqr_blocks1 = blocks1;
    sqr_blocks2 = blocks2;
end