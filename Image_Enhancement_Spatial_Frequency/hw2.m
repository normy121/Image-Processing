clc;
close all;
clear;

moon_img = double(imread('HW2_test_image/blurry_moon.tif'));
skeleton_img = double(imread('HW2_test_image/skeleton_orig.bmp'));

A = 1.7;
mask_1 = [0 1 0; 1 -4 1; 0 1 0];
mask_2 = [1 1 1; 1 -8 1; 1 1 1];

%Spatial Domain
Laplacian_img = uint8(myLaplacian(moon_img, mask_1));
moon_img = uint8(moon_img);

unsharp = moon_img.*1-Laplacian_img;
high_boost = moon_img.*A-Laplacian_img;

f = figure('Name','Moon Spatial Domain','NumberTitle','off');
f.WindowState = 'maximized';
subplot(241);
imshow(moon_img);
title('Original Image', 'FontSize', 16);
subplot(242);
imshow(Laplacian_img);
title('First Laplacian Operator Image', 'FontSize', 16);
subplot(243);
imshow(unsharp);
title('Unsharp Masking Image', 'FontSize', 16);
subplot(244);
imshow(high_boost);
title('High-boost Sharpened Image (A = 1.7)', 'FontSize', 16);

Laplacian_img = uint8(myLaplacian(double(moon_img), mask_2));
moon_img = uint8(moon_img);

unsharp = moon_img.*1-Laplacian_img;
high_boost = moon_img.*A-Laplacian_img;

subplot(245);
imshow(moon_img);
title('Original Image', 'FontSize', 16);
subplot(246);
imshow(Laplacian_img);
title('Second Laplacian Operator Image', 'FontSize', 16);
subplot(247);
imshow(unsharp);
title('Unsharp Masking Image', 'FontSize', 16);
subplot(248);
imshow(high_boost);
title('High-boost Sharpened Image', 'FontSize', 16);

Laplacian_img = uint8(myLaplacian(skeleton_img, mask_1));
skeleton_img = uint8(skeleton_img);

unsharp = skeleton_img.*1-Laplacian_img;
high_boost = skeleton_img.*A-Laplacian_img;

f = figure('Name','Skeleton Spatial Domain','NumberTitle','off');
f.WindowState = 'maximized';
subplot(241);
imshow(skeleton_img);
title('Original Image', 'FontSize', 16);
subplot(242);
imshow(Laplacian_img);
title('First Laplacian Operator Image', 'FontSize', 16);
subplot(243);
imshow(unsharp);
title('Unsharp Masking Image', 'FontSize', 16);
subplot(244);
imshow(high_boost);
title('High-boost Sharpened Image (A = 1.7)', 'FontSize', 16);

Laplacian_img = uint8(myLaplacian(double(skeleton_img), mask_2));
skeleton_img = uint8(skeleton_img);

unsharp = skeleton_img.*1-Laplacian_img;
high_boost = skeleton_img.*A-Laplacian_img;

subplot(245);
imshow(skeleton_img);
title('Original Image', 'FontSize', 16);
subplot(246);
imshow(Laplacian_img);
title('Second Laplacian Operator Image', 'FontSize', 16);
subplot(247);
imshow(unsharp);
title('Unsharp Masking Image', 'FontSize', 16);
subplot(248);
imshow(high_boost);
title('High-boost Sharpened Image (A = 1.7)', 'FontSize', 16);

%Frequency Domain
PQ = paddedsize(size(moon_img));
mask_f1 = fft2(double(mask_1), PQ(1), PQ(2));
image_f1 = fft2(double(moon_img), PQ(1), PQ(2));
fft_result = mask_f1 .* image_f1;
fft_result = ifft2(fft_result);
fft_result = uint8(fft_result(1:size(moon_img,1),1:size(moon_img,2)));
unsharp = moon_img.*1-fft_result;
high_boost = moon_img.*A-fft_result;

f = figure('Name','Moon Frequency Domain','NumberTitle','off');
f.WindowState = 'maximized';
subplot(241);
imshow(moon_img);
title('Original Image', 'FontSize', 16);
subplot(242);
imshow(fft_result);
title('First Laplacian Operator Image', 'FontSize', 16);
subplot(243);
imshow(unsharp);
title('Unsharp Masking Image', 'FontSize', 16);
subplot(244);
imshow(high_boost);
title('High-boost Sharpened Image (A = 1.7)', 'FontSize', 16);

PQ = paddedsize(size(moon_img));
mask_f2 = fft2(double(mask_2), PQ(1), PQ(2));
image_f2 = fft2(double(moon_img), PQ(1), PQ(2));
fft_result = mask_f2 .* image_f2;
fft_result = ifft2(fft_result);
fft_result = uint8(fft_result(1:size(moon_img,1),1:size(moon_img,2)));
unsharp = moon_img.*1-fft_result;
high_boost = moon_img.*A-fft_result;

subplot(245);
imshow(moon_img);
title('Original Image', 'FontSize', 16);
subplot(246);
imshow(fft_result);
title('Second Laplacian Operator Image', 'FontSize', 16);
subplot(247);
imshow(unsharp);
title('Unsharp Masking Image', 'FontSize', 16);
subplot(248);
imshow(high_boost);
title('High-boost Sharpened Image (A = 1.7)', 'FontSize', 16);

PQ = paddedsize(size(skeleton_img));
mask_f1 = fft2(double(mask_1), PQ(1), PQ(2));
image_f1 = fft2(double(skeleton_img), PQ(1), PQ(2));
fft_result = mask_f1 .* image_f1;
fft_result = ifft2(fft_result);
fft_result = uint8(fft_result(1:size(skeleton_img,1),1:size(skeleton_img,2)));
unsharp = skeleton_img.*1-fft_result;
high_boost = skeleton_img.*A-fft_result;

f = figure('Name','Skeleton Frequency Domain','NumberTitle','off');
f.WindowState = 'maximized';
subplot(241);
imshow(skeleton_img);
title('Original Image', 'FontSize', 16);
subplot(242);
imshow(fft_result);
title('First Laplacian Operator Image', 'FontSize', 16);
subplot(243);
imshow(unsharp);
title('Unsharp Masking Image', 'FontSize', 16);
subplot(244);
imshow(high_boost);
title('High-boost Sharpened Image (A = 1.7)', 'FontSize', 16);

PQ = paddedsize(size(skeleton_img));
mask_f2 = fft2(double(mask_2), PQ(1), PQ(2));
image_f2 = fft2(double(skeleton_img), PQ(1), PQ(2));
%mask_f2 = fftshift(mask_f2);
%image_f2 = fftshift(image_f2);
fft_result = mask_f2 .* image_f2;
%fft_result = ifftshift(fft_result);
fft_result = ifft2(fft_result);
fft_result = uint8(fft_result(1:size(skeleton_img,1),1:size(skeleton_img,2)));
unsharp = skeleton_img.*1-fft_result;
high_boost = skeleton_img.*A-fft_result;

subplot(245);
imshow(skeleton_img);
title('Original Image', 'FontSize', 16);
subplot(246);
imshow(fft_result);
title('Second Laplacian Operator Image', 'FontSize', 16);
subplot(247);
imshow(unsharp);
title('Unsharp Masking Image', 'FontSize', 16);
subplot(248);
imshow(high_boost);
title('High-boost Sharpened Image (A = 1.7)', 'FontSize', 16);

%High Boost Filtering Comparison
f = figure('Name','High Boost Filtering','NumberTitle','off');
f.WindowState = 'maximized';
Laplacian_img = uint8(myLaplacian(double(moon_img), mask_1));
subplot(241);
imshow(moon_img.*1-Laplacian_img);
title('Original Moon (A = 1.0)', 'FontSize', 16);
subplot(242);
imshow(moon_img.*1.3-Laplacian_img);
title('High Boost Moon (A = 1.3)', 'FontSize', 16);
subplot(243);
imshow(moon_img.*1.7-Laplacian_img);
title('High Boost Moon (A = 1.7)', 'FontSize', 16);
subplot(244);
imshow(moon_img.*2.0-Laplacian_img);
title('High Boost Moon (A = 2.0)', 'FontSize', 16);
Laplacian_img = uint8(myLaplacian(double(skeleton_img), mask_1));
subplot(245);
imshow(skeleton_img.*1-Laplacian_img);
title('Original Skeleton (A = 1.0)', 'FontSize', 16);
subplot(246);
imshow(skeleton_img.*1.3-Laplacian_img);
title('High Boost Skeleton (A = 1.3)', 'FontSize', 16);
subplot(247);
imshow(skeleton_img.*1.7-Laplacian_img);
title('High Boost Skeleton (A = 1.7)', 'FontSize', 16);
subplot(248);
imshow(skeleton_img.*2.0-Laplacian_img);
title('High Boost Skeleton (A = 2.0)', 'FontSize', 16);

function [output_img] = myLaplacian(input_img, mask)
    output_img = zeros(size(input_img));
    [row,col] = size(input_img);
    for i = 1 : row - 2
        for j = 1 : col - 2
            output_img(i,j) = sum(sum(mask.*input_img(i:i+2,j:j+2)));    
        end
    end
end

function PQ = paddedsize(AB, CD, ~)
    if nargin == 1
        PQ = 2 * AB;
    elseif nargin == 2 && ~ischar(CD)
        PQ = AB + CD - 1;
        PQ = 2 * ceil(PQ / 2);
    elseif nargin == 2
        m = max(AB); % Maximum dimension.
        P = 2^nextpow2(2*m); % Find power-of-2 at least twice m.
        PQ = [P, P];
    elseif nargin == 3
        m = max([AB CD]); %Maximum dimension.
        P = 2^nextpow2(2 * m);
        PQ = [P, P];
    else
        error('Wrong number of inputs.')
    end
end