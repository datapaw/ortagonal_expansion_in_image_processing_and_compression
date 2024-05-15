%% Image sampling without map
% Importing the image first method
Image = imread('images/lena_256.jpg');
figure;imshow(Image)

% Convert the image to grayscale if necessary
if size(Image, 3) == 3
    Image= rgb2gray(Image);    figure;imshow(Image)
    ...imwrite(Image,"images/grayed.jpg")
end

%% Image sampling with map (program to convert to normal image)
% Importing the image second method
% [inpict, map] = imread('images/cameraman_map.png');
% figure;imshow(inpict, map)
% Image = ind2gray(inpict, map);
% figure;imshow(Image)
...imwrite(Image,"images/cameraman.png")

%% Downsampling and other simple solutions
% Downsampling by how many times
DS = 3;

% Obtaining the size of the image using:
% size(rows/column/page)
% Row - number of rows in the image / height
% Column - number of columns / width
% Page - encodes the three color channels RGB
[Height, Width, RGB] = size(Image);

% Sampling calculations
% DS - downsampling
DSimage_1 = Image(1:DS:Height, 1:DS:Width);
figure; imshow(DSimage_1); title(["Downsampled by" num2str(DS)])
...imwrite(DSimage_1,"images/DSimage_1.jpg")

% Downsampling and resizing up
% RU - resizing up
DSRUimage_1 = imresize(DSimage_1,[Height,Width]);
figure;imshow(DSRUimage_1); title("Downsampling and resizing up")
...imwrite(DSRUimage_1,"images/DSRUimage_1.jpg")

% Downsampling and resizing up with filter
% RUF - resizing up + filter
RUFimage_1 = imfilter(DSRUimage_1, fspecial("laplacian"));
figure
subplot(2,1,1);imshow(RUFimage_1); title("Downsampling and resizing up with laplacian filter")

RUFimage_2 = imfilter(DSRUimage_1, fspecial("gaussian",6,2));
subplot(2,1,2);imshow(RUFimage_2); title("Downsampling and resizing up with gaussian filter")
...imwrite(RUFimage_2,"images/RUFimage_2.jpg")

%% Uniform Quantizer
% Desired number of bits for quantization (4-64)
% 4|8|16|32|64
UQ_bits = 16;

% Convert the image to double precision for calculations
UQ_values = double(Image);

% Flatten the image into a 1D array
UQ_input_values = UQ_values(:);

% Compute quantization step size
UQ_step_size = (max(UQ_input_values) - min(UQ_input_values)) / (UQ_bits - 1);

% Compute quantization intervals
UQ_quantization_intervals = min(UQ_input_values):UQ_step_size:max(UQ_input_values);

% Quantize input values based on quantization intervals
UQ_quantized_values = zeros(size(UQ_input_values));
for i = 1:UQ_bits
    % Assign input values to quantization levels based on intervals
    UQ_quantized_values(UQ_input_values >= UQ_quantization_intervals(i)) = i - 1;
end

% Reshape quantized values back to image dimensions
UQimage = reshape(UQ_quantized_values, size(Image));

% Display original and quantized images
figure;subplot(1, 2, 1);imshow(uint8(Image));title('Original Image');
subplot(1, 2, 2);imshow(uint8(UQimage * (255 / (UQ_bits - 1))));title(['Quantized Image (Uniform, ', num2str(UQ_bits), ' levels)']);
...imwrite(uint8(UQimage),"images/UQimage.jpg")

%% Lloyd-Max Quantizer
% Desired number of bits for quantization (1-8)
% 2|4|8|16|32|64|128|256
LMQ_bits = 4;

% Convert the image to double precision for calculations
LMQ_values = double(Image);

% Flatten the image into a 1D array
LMQ_input_values = LMQ_values(:);

% Initialize centroids based on the desired number of quantization levels
LMQ_min_value = min(LMQ_input_values);
LMQ_max_value = max(LMQ_input_values);
LMQ_centroids = linspace(LMQ_min_value, LMQ_max_value, LMQ_bits);

% Iterative Lloyd-Max algorithm
max_iterations = 128;
for iter = 1:max_iterations
    % Assign each input value to the nearest centroid

    % calculates the minimum absolute difference between each element
    % in LMQ_input_values and the array centroids, along the second 
    % dimension, effectively assigning each element of LMQ_input_values 
    % to the nearest centroid
    [~, LMQ_index] = min(abs(LMQ_input_values - LMQ_centroids), [], 2);
    
    % Update centroids to be the mean of their assigned values
    for i = 1:length(LMQ_centroids)
        
        % computes the mean of the elements in the array LMQ_input_values
        % that are assigned to the centroid represented by index i
        LMQ_centroids(i) = mean(LMQ_input_values(LMQ_index == i));
    end
    
    % Check for convergence, checking if matrix end
    if iter > 1 && isequal(LMQ_centroids, LMQ_prev_centroids)
        break;
    end
    LMQ_prev_centroids = LMQ_centroids;
end

% Assign each input value to its nearest centroid
[~, LMQ_quantized_values] = min(abs(LMQ_input_values - LMQ_centroids), [], 2);

% Quantized image
LMQimage = reshape(LMQ_centroids(LMQ_quantized_values), size(Image));

% Display the original and quantized images
figure; subplot(1, 2, 1); imshow(uint8(LMQ_values)); title('Original Image');
subplot(1, 2, 2); imshow(uint8(LMQimage)); title(['Quantized Image (' num2str(LMQ_bits) ' bits)']);
...imwrite(uint8(LMQimage),"images/LMQimage.jpg")

%% NonUniform Quantizer
% Desired number of bits for quantization (2-384)
% 32|64|128|256|384
NU_bits = 32;

% Convert the image to double precision for calculations
NU_values = double(Image);

% Flatten the image into a 1D array
NU_input_values = NU_values(:);

% Initialize centroids based on the desired number of quantization levels
NU_min_value = min(NU_input_values);
NU_max_value = max(NU_input_values);

% Compute histogram of input image
[counts, bins] = imhist(uint8(NU_values));
...figure;plot(bins, counts);title('Histogram of inputs');

% Compute cumulative histogram
NU_cumulative_counts = cumsum(counts);

% Compute cumulative distribution function (CDF)
NU_cdf = NU_cumulative_counts / numel(NU_input_values);
...figure;plot(NU_cdf);title('Cumulative colour distribution');

% Without the "NU_cdf = NU_cdf + eps * rand(size(NU_cdf));" line there will
% be an error: "Sample points must be unique" suggests 
% that there are duplicate values in the cdf array, which
% is used as the sample points for interpolation.

% Add small noise to ensure uniqueness in CDF
epsilon = 1e-10;
NU_cdf = NU_cdf + epsilon * rand(size(NU_cdf));

% Compute quantization intervals based on CDF
NU_quantization_intervals = interp1(NU_cdf, bins, linspace(NU_min_value/NU_max_value, NU_max_value/NU_max_value, NU_bits));

% Quantize input values based on quantization intervals
NU_quantized_values = zeros(size(NU_input_values));
for i = 1:(NU_bits - 1)
    % This loop quantizes the input pixel values based on the quantization
    % intervals computed earlier. It assigns each input value to the 
    % corresponding quantization level based on the intervals. The loop
    % iterates over each interval except the last one (handled separately)
    % and assigns the quantization level i to the input values falling 
    % within the interval 
    
    % Assign input values to quantization levels based on intervals
    NU_quantized_values(NU_input_values >= NU_quantization_intervals(i) & NU_input_values < NU_quantization_intervals(i+1)) = i;
end
    % Handle the last interval separately to avoid out-of-bounds indexing
    NU_quantized_values(NU_input_values >= NU_quantization_intervals(NU_bits)) = NU_bits;

% Reshape quantized values back to image dimensions
NUimage = reshape(NU_quantized_values, size(NU_values));

% Display original and quantized images
figure;subplot(1, 2, 1);imshow(uint8(Image));title('Original Image');
subplot(1, 2, 2);imshow(uint8(NUimage));title(['Quantized Image (Non-uniform, ', num2str(NU_bits), ' levels)']);
...imwrite(uint8(NUimage),"images/NUimage.jpg")

%% ###################################
%  #        Image transforms         #
%  ###################################
%% Haar wavelet
% BOOK: https://en.wikipedia.org/wiki/Haar_wavelet
% Perform the Haar transformation
[LL, LH, HL, HH, HaarImage] = haar_wavelet(Image);

figure;
subplot(2,2,1);imshow(uint8(LL));title('LL')
subplot(2,2,2);imshow(uint8(HL));title('HL')
subplot(2,2,3);imshow(uint8(LH));title('LH')
subplot(2,2,4);imshow(uint8(HH));title('HH')
title('Haar images separated')
figure;imshow(uint8(HaarImage));title('Haar image')

% Test of inside function
% [LL, HL, LH, HH] = dwt2(Image, 'haar');
% figure; 
% subplot(2,2,1);imshow(uint8(LL));title('LL')
% subplot(2,2,2);imshow(uint8(HL));title('HL')
% subplot(2,2,3);imshow(uint8(LH));title('LH')
% subplot(2,2,4);imshow(uint8(HH));title('HH')

%% Daubeches wavelet
% Perform the Daubechies transformation
[LL, LH, HL, HH, DaubechiesImage] = daubechies_wavelet(Image, 2);% 2 albo 3

figure;
subplot(2,2,1);imshow(uint8(LL));title('LL')
subplot(2,2,2);imshow(uint8(HL));title('HL')
subplot(2,2,3);imshow(uint8(LH));title('LH')
subplot(2,2,4);imshow(uint8(HH));title('HH')
title('Daubechies images separated')
figure;imshow(uint8(DaubechiesImage));title('Daubechies image')

%% Coiflet wavelet
% Perform the Coiflet transformation
[LL, LH, HL, HH, CoifletImage] = coiflet_wavelet(Image, 2);% 1 albo 2

figure;
subplot(2,2,1);imshow(uint8(LL));title('LL')
subplot(2,2,2);imshow(uint8(HL));title('HL')
subplot(2,2,3);imshow(uint8(LH));title('LH')
subplot(2,2,4);imshow(uint8(HH));title('HH')
title('Coiflet images separated')
figure;imshow(uint8(CoifletImage));title('Coiflet image')

%% Symlet wavelet
% Perform the Symlet transformation
[LL, LH, HL, HH, SymletImage] = symlet_wavelet(Image, 2);% 2 albo 3

figure;
subplot(2,2,1);imshow(uint8(LL));title('LL')
subplot(2,2,2);imshow(uint8(HL));title('HL')
subplot(2,2,3);imshow(uint8(LH));title('LH')
subplot(2,2,4);imshow(uint8(HH));title('HH')
title('Coiflet images separated')
figure;imshow(uint8(SymletImage));title('Symlet image')

%% Discrete Cosine Transform - Basic MATLAB
mat_dct = dct2(Image);
mat_idct = idct2(mat_dct);

figure('Name','DCT MATLAB','NumberTitle','off');
subplot(1, 3, 1);
imshow(Image);
title('Original Image');

subplot(1, 3, 2);
imshow(mat_dct);
title('DCT Transform');

subplot(1, 3, 3);
imshow(uint8(mat_idct));
title('Reconstructed Image');

%% DCT Compression - MATLAB

% Perform Discrete Cosine Transform (DCT) on the input image
J = dct2(Image);

% Display the logarithm of the absolute values of DCT coefficients
figure('Name','DCT','NumberTitle','off');
imshow(log(abs(J)),[])
colormap parula
colorbar

% Thresholding: Set small magnitude DCT coefficients to zero
J(abs(J) < 10) = 0;

% Reconstruct the image using the inverse DCT (IDCT)
K = idct2(J);

% Rescale the reconstructed image to ensure pixel values are in the proper range
K = rescale(K);

% Display the original grayscale image and the processed image side by side
figure('Name','Processed','NumberTitle','off');
montage({Image,K})
title('Original Grayscale Image (Left) and Processed Image (Right)');

%% DCT Compression - Transformation matrix approach
% https://www.youtube.com/watch?v=mUKPy3r0TTI

[filename, pathname] = uigetfile('images/*.*', 'Select Grayscale Image');
filpath = strcat(pathname,filename)
img = imread(filpath);

% if size(img, 3) == 3
%     img= rgb2gray(img);
% end

C = dct2(double(img));

% Visualize DCT coefficients
figure
subplot(1,2,1)
imshow(log(abs(C)), [])
title('DCT Coefficients before truncation')
colormap(gca, jet(64))
colorbar

C(abs(C) < 10) = 0;

subplot(1,2,2)
imshow(log(abs(C)), [])
title('DCT Coefficients after truncation')
colormap(gca, jet(64))
colorbar

Ct = idct2(C);

% Save original and compressed images
imwrite(uint8(img), 'DCT/Original_Image.jpg', 'Quality', 100);
imwrite(uint8(Ct), 'DCT/DCT_Compressed_Image.jpg', 'Quality', 100);

% Display original and compressed images side by side
figure
imshowpair(img, Ct, 'montage')
title('Original Image (Left) and Compressed Image (Right)')

% Compute absolute difference between original and compressed images
error_image = abs(double(img) - double(Ct));

% Display the error image
figure;
imshow(error_image, []);
title('Error Image');

%% DCT Compression - DCT Transformation matrix (T)
% https://www.youtube.com/watch?v=mUKPy3r0TTI

[filename, pathname] = uigetfile('images/*.*', 'Select Grayscale Image');
img = double(imread(strcat(pathname,filename)));

T = dctmtx(8);
xdct = @(block_struct) T * (block_struct.data) * T';
C = blockproc(img, [8 8], xdct);

mask = [1 1 1 1 0 0 0 0
        1 1 1 0 0 0 0 0
        1 1 0 0 0 0 0 0
        1 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0];

% mask = [1 1 1 1 1 1 1 1
%         1 1 1 1 1 1 1 0
%         1 1 1 1 1 1 0 0
%         1 1 1 1 1 0 0 0
%         1 1 1 1 0 0 0 0
%         1 1 1 0 0 0 0 0
%         1 1 0 0 0 0 0 0
%         1 0 0 0 0 0 0 0];

% mask = zeros(8,8);
% mask = ones(8,8);

Ct = blockproc(C, [8 8], @(block_struct) (mask .* block_struct.data));

invdct = @(block_struct) T' * (block_struct.data) * T;
invC = blockproc(Ct, [8 8], invdct);

% Save original and compressed images
imwrite(uint8(img), 'DCT/Original_Image.jpg', 'Quality', 100);
imwrite(uint8(invC), 'DCT/DCT_Compressed_Image.jpg', 'Quality', 100);

% Display original and compressed images side by side
figure
imshowpair(img, invC, 'montage')
title('Original Image (Left) and Compressed Image (Right)')

% Compute absolute difference between original and compressed images
error_image = abs(double(img) - double(invC));

% Display the error image
figure;
imshow(error_image, []);
title('Error Image');

%% Discrete Cosine Transform - Extended

% Select an image file
[filename, pathname] = uigetfile('images/*.*', 'Select Image');

% Start timer
tic

% Read the selected image
image = imread(strcat(pathname,filename));

% Convert the image to grayscale if it's an RGB image
if size(image, 3) == 3
    image = rgb2gray(image);
end

% Import an image
% image = Image

% Convert the image data type to double for processing
image = double(image);

% Define the block size for processing
block_size = 16;

% Visualize the image with block boundaries
block_visualization(uint8(image), block_size);

% Perform 2D Discrete Cosine Transform (DCT)
dct_image = my_dct2_block(double(image), block_size);

% Visualize the DCT coefficients before compression
figure
subplot(1,2,1)
log_dct_image = log(abs(dct_image) + 1); % Adding 1 to avoid log(0)
imshow(log_dct_image, []);
colormap('jet') % Use the 'jet' colormap for better visualization
colorbar; % Add a color bar for reference
title('DCT Before Compression');

% Compress the DCT coefficients
% compression_ratio = 0.96;
% dct_image_compressed = compression(dct_image, compression_ratio);
% dct_image_compressed = compression2(dct_image, block_size, compression_ratio);
dct_image_compressed = compression3(dct_image, 4); % Adjust compression ratio as needed

% Visualize the compressed DCT coefficients
subplot(1,2,2)
log_dct_image_c = log(abs(dct_image_compressed) + 1); % Adding 1 to avoid log(0)
imshow(log_dct_image_c, []);
colormap('jet') % Use the 'jet' colormap for better visualization
colorbar; % Add a color bar for reference
title('DCT After Compression');

% Reconstruct the image using inverse DCT
inv = my_idct2_block(dct_image_compressed, block_size);

% Save the original and compressed images
imwrite(uint8(image), 'DCT/Original_Image.jpg', 'Quality', 100);
imwrite(uint8(inv), 'DCT/DCT_Compressed_Image.jpg', 'Quality', 100);

% Compute the sizes of the original and compressed images
original_kB = dir('DCT/Original_Image.jpg').bytes / 1024
compressed_kB = dir('DCT/DCT_Compressed_Image.jpg').bytes / 1024

% Display the original and compressed images side by side
figure
imshowpair(uint8(image), uint8(inv), 'montage')
title_text = sprintf('Original Image, Size: %.2f kB (Left) and Compressed Image, Size: %.2f kB (Right)', original_kB, compressed_kB);
title(title_text);

% Compute the absolute difference between original and compressed images
error_image = abs(double(image) - double(inv));

% Display the error image
figure;
imshow(error_image, []);
title('Error Image');

% End timer and display elapsed time
toc

%% Discrete fourier transform

% Select an image file
[filename, pathname] = uigetfile('images/*.*', 'Select Image');

% Start timer
tic

% % Read the selected image
% image = imread(strcat(pathname,filename));
% figure
% imshow(image)
% % Perform edge detection
% edge_image = edge_detection(image);

% % Display the original image and the edge-detected image side by side
% figure;
% subplot(1,2,1);
% imshow(image);
% title('Original Image');
% subplot(1,2,2);
% imshow(edge_image);
% title('Edge-Detected Image');

% Convert the image to grayscale if it's an RGB image
if size(image, 3) == 3
    image = rgb2gray(image);
end

% % Display the original grayscale image
% figure;
% imshow(image);
% title('Original Grayscale Image');

% Convert the image data type to double for processing
image = double(image);

% Define the block size for processing
block_size = 32;

% Visualize the image with block boundaries
block_visualization(uint8(image), block_size);

% Perform 2D Discrete Cosine Transform (DCT)
dct_image = my_dft(double(image), block_size);

figure
log_dct_image = log(abs(dct_image) + 1); % Adding 1 to avoid log(0)
imshow(log_dct_image, []);

% Reconstruct the image using inverse DCT
inv = my_idft(dct_image, block_size);
figure
imshow(uint8(inv))
% % Display the original and compressed images side by side
% figure
% imshowpair(uint8(image), uint8(inv), 'montage')
% title('Original Image (Left) and Reconstructed Image (Right)');

% Compute the absolute difference between original and compressed images
error_image = abs(double(image) - double(inv));

% Display the error image
figure;
imshow(error_image, []);
title('Error Image');

% % Calculate the size of the original grayscale image
% original_size_bytes = numel(image);
% 
% % Calculate the size of the reconstructed image after inverse DFT
% reconstructed_size_bytes = numel(uint8(inv));
% 
% % Display the sizes
% fprintf('Original Image Size: %d bytes\n', original_size_bytes);
% fprintf('Reconstructed Image Size: %d bytes\n', reconstructed_size_bytes);
% 

% End timer and display elapsed time
toc

%% Edge detection using Sobel & Canny filters
% Read the image
% Select an image file
[filename, pathname] = uigetfile('images/*.*', 'Select Image');

% Read the selected image
image = imread(strcat(pathname,filename));

% Convert to grayscale if necessary
if size(image, 3) == 3
    image = rgb2gray(image);
end

I = image;
BW1 = edge(I,'sobel');
BW2 = edge(I,'canny');
tiledlayout(1,2)

nexttile
imshow(BW1)
title('Sobel Filter')

nexttile
imshow(BW2)
title('Canny Filter')
