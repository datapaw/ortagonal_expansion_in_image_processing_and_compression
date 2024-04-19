function [LL, LH, HL, HH, compressed_image] = daubechies_wavelet(image, wavelet_order)
    % Convert the input image to double precision
    image = double(image);

    % Get the size of the input image
    [rows, cols] = size(image);

    % Initialize the compressed image matrix
    compressed_image = zeros(rows, cols);

    % Define Daubechies filter coefficients manually
    [Lo_D, Hi_D] = daubechies_filter(wavelet_order);

    % Perform the Daubechies wavelet transform on each row
    for i = 1:rows
        compressed_image(i, :) = daubechies_transform_1D(image(i, :), Lo_D, Hi_D);
    end

    % Perform the Daubechies wavelet transform on each column
    for j = 1:cols
        compressed_image(:, j) = daubechies_transform_1D(compressed_image(:, j)', Lo_D, Hi_D);
    end

    LL = compressed_image(1:(rows/2), 1:(cols/2));
    LH = compressed_image(1:(rows/2+1), (cols/2+1):cols);
    HL = compressed_image((rows/2+1):rows, 1:(cols/2+1));
    HH = compressed_image((rows/2+1):rows, (cols/2+1):cols);
end

function [Lo_D, Hi_D] = daubechies_filter(wavelet_order)
    % Define Daubechies filter coefficients manually
    switch wavelet_order
        case 2
            Lo_D = [0.48296, 0.83651, 0.22414, -0.12940];
            Hi_D = [-0.12940, -0.22414, 0.83651, -0.48296];
        case 3
            Lo_D = [0.33267, 0.80689, 0.45988, -0.13501, -0.08544, 0.03523];
            Hi_D = [-0.03523, -0.08544, 0.13501, 0.45988, -0.80689, 0.33267];
        otherwise
            error('Unsupported Daubechies wavelet order');
    end
end

function transformed_signal = daubechies_transform_1D(signal, Lo_D, Hi_D)
    % Get the length of the input signal
    N = length(signal);

    % Initialize the transformed signal vector
    transformed_signal = zeros(1, N);

    % Daubechies wavelet transform
    for k = 1:N/2
        for m = 1:length(Lo_D)
            idx = 2*k - 1 - length(Lo_D) + m;
            if idx < 1
                idx = idx + N;
            end
            transformed_signal(k) = transformed_signal(k) + Lo_D(m) * signal(idx);
            transformed_signal(k+N/2) = transformed_signal(k+N/2) + Hi_D(m) * signal(idx);
        end
    end
end
