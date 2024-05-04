function dct_image_compressed = compression3(dct_image, threshold)
    % Get the size of the DCT image
    [rows, cols] = size(dct_image);

    % Initialize the compressed DCT image
    dct_image_compressed = zeros(rows, cols);

    % Apply thresholding
    for i = 1:rows
        for j = 1:cols
            if log(abs(dct_image(i, j))) < threshold
                dct_image_compressed(i, j) = 0; % Set small magnitude DCT coefficients to zero
            else
                dct_image_compressed(i, j) = dct_image(i, j); % Keep the coefficient unchanged
            end
        end
    end
end
