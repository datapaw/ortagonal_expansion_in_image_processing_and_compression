function [LL, LH, HL, HH, compressed_image] = coiflet_wavelet(image, wavelet_order)
    % Convert the input image to double precision
    image = double(image);

    % Get the size of the input image
    [rows, cols] = size(image);

    % Initialize the compressed image matrix
    compressed_image = zeros(rows, cols);

    % Define Coiflet filter coefficients manually
    [Lo_D, Hi_D] = coiflet_filter(wavelet_order);

    % Perform the Coiflet wavelet transform on each row
    for i = 1:rows
        compressed_image(i, :) = coiflet_transform_1D(image(i, :), Lo_D, Hi_D);
    end

    % Perform the Coiflet wavelet transform on each column
    for j = 1:cols
        compressed_image(:, j) = coiflet_transform_1D(compressed_image(:, j)', Lo_D, Hi_D);
    end

    LL = compressed_image(1:(rows/2), 1:(cols/2));
    LH = compressed_image(1:(rows/2+1), (cols/2+1):cols);
    HL = compressed_image((rows/2+1):rows, 1:(cols/2+1));
    HH = compressed_image((rows/2+1):rows, (cols/2+1):cols);
end

function [Lo_D, Hi_D] = coiflet_filter(wavelet_order)
    % Define Coiflet filter coefficients manually
    switch wavelet_order
        case 1
            Lo_D = [-1/8, 1/4, 3/4, 1/4, -1/8];
            Hi_D = [0, 0.5, -1, 0.5, 0];
        case 2
            Lo_D = [-0.0157, 0.0313, 0.1329, 0.4869, 0.7567, 0.4116, -0.2298, -0.1601];
            Hi_D = [0.1601, -0.2298, -0.4116, 0.7567, -0.4869, 0.1329, -0.0313, -0.0157];
        otherwise
            error('Unsupported Coiflet wavelet order');
    end
end

function transformed_signal = coiflet_transform_1D(signal, Lo_D, Hi_D)
    % Get the length of the input signal
    N = length(signal);

    % Initialize the transformed signal vector
    transformed_signal = zeros(1, N);

    % Coiflet wavelet transform
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
