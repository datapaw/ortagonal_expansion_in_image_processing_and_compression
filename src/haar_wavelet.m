function [LL, LH, HL, HH, compressed_image] = haar_wavelet(image)
    % Convert the input image to double precision
    image = double(image);

    % Get the size of the input image
    [rows, cols] = size(image);

    % Initialize the compressed image matrix
    compressed_image = zeros(rows, cols);

    % Perform the Haar transform on each row
    for i = 1:rows
        compressed_image(i, :) = haar_transform_1D(image(i, :));
    end

    % Perform the Haar transform on each column
    for j = 1:cols
        compressed_image(:, j) = haar_transform_1D(compressed_image(:, j)');
    end

    LL = compressed_image(1:(rows/2),1:(cols/2),:);
    LH = compressed_image(1:(rows/2+1),(cols/2+1):cols,:);
    HL = compressed_image((rows/2+1):rows,1:(cols/2+1),:);
    HH = compressed_image((rows/2+1):rows,(cols/2+1):cols,:);

end

function transformed_signal = haar_transform_1D(signal)
    % Get the length of the input signal
    N = length(signal);

    % Initialize the transformed signal vector
    transformed_signal = zeros(1, N);

    % Haar wavelet transform
    for k = 1:N/2
        transformed_signal(k) = (signal(2*k-1) + signal(2*k)) / sqrt(2);
        transformed_signal(k+N/2) = (signal(2*k-1) - signal(2*k)) / sqrt(2);
    end
end