%% Image ideal sampling
%importing the image
Image = imread("images/cat.jpg");
figure;imshow(Image)

% Convert the image to grayscale if necessary
if size(Image, 3) == 3
    Image= rgb2gray(Image);
end
figure;imshow(Image)
...imwrite(Image,"images/grayed.jpg")

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

%% Image transforms










