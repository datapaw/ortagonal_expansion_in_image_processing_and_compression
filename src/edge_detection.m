function edge_image = edge_detection(image)
    % Convert to grayscale if necessary
    if size(image, 3) == 3
        image = rgb2gray(image);
    end

    % Compute Fourier transform of the image
    F = fft2(image);

    % Shift zero frequency component to the center
    F_shifted = fftshift(F);

    % Define high-pass filter (Laplacian filter)
    [rows, cols] = size(image);
    [X, Y] = meshgrid(1:cols, 1:rows);
    center_x = floor(cols / 2) + 1;
    center_y = floor(rows / 2) + 1;
    laplacian_filter = -4 * pi^2 * ((X - center_x).^2 + (Y - center_y).^2);
    laplacian_filter(center_y, center_x) = laplacian_filter(center_y, center_x) + 1;

    % Apply high-pass filter in frequency domain
    F_filtered = F_shifted .* laplacian_filter;

    % Shift back to original position
    F_unshifted = ifftshift(F_filtered);

    % Inverse Fourier transform
    edge_image = ifft2(F_unshifted);

    % Take absolute value and rescale to [0, 255]
    edge_image = abs(edge_image);
    edge_image = uint8(edge_image / max(edge_image(:)) * 255);
end