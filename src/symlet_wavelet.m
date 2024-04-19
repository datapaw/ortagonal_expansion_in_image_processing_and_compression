function [LL, LH, HL, HH, compressed_image] = symlet_wavelet(image, wavelet_order)
    % Convert the input image to double precision
    image = double(image);

    % Get the size of the input image
    [rows, cols] = size(image);

    % Initialize the compressed image matrix
    compressed_image = zeros(rows, cols);

    % Define Symlet filter coefficients manually
    [Lo_D, Hi_D] = symlet_filter(wavelet_order);

    % Perform the Symlet wavelet transform on each row
    for i = 1:rows
        compressed_image(i, :) = symlet_transform_1D(image(i, :), Lo_D, Hi_D);
    end

    % Perform the Symlet wavelet transform on each column
    for j = 1:cols
        compressed_image(:, j) = symlet_transform_1D(compressed_image(:, j)', Lo_D, Hi_D);
    end

    LL = compressed_image(1:(rows/2), 1:(cols/2));
    LH = compressed_image(1:(rows/2+1), (cols/2+1):cols);
    HL = compressed_image((rows/2+1):rows, 1:(cols/2+1));
    HH = compressed_image((rows/2+1):rows, (cols/2+1):cols);
end

function [Lo_D, Hi_D] = symlet_filter(wavelet_order)
    % Define Symlet filter coefficients manually
    switch wavelet_order
        case 2
            Lo_D = [-0.1294, 0.2241, 0.8365, 0.4829];
            Hi_D = [-0.4829, 0.8365, -0.2241, -0.1294];
        case 3
            Lo_D = [-0.0352, -0.0854, 0.1350, 0.4599, 0.8069, 0.3327];
            Hi_D = [-0.3327, 0.8069, -0.4599, -0.1350, 0.0854, 0.0352];
        otherwise
            error('Unsupported Symlet wavelet order');
    end
end

function transformed_signal = symlet_transform_1D(signal, Lo_D, Hi_D)
    % Get the length of the input signal
    N = length(signal);

    % Initialize the transformed signal vector
    transformed_signal = zeros(1, N);

    % Symlet wavelet transform
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
