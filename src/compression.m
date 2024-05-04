function dct_image_compressed = compression(dct_image, compression_ratio)
    % Flatten the DCT coefficients into a vector
    dct_coeffs = dct_image(:);
    
    % Sort the coefficients by magnitude
    [~, sorted_indices] = sort(abs(dct_coeffs), 'descend');
    
    % Calculate the threshold index based on the compression ratio
    num_coeffs = numel(dct_coeffs);
    threshold_index = ceil(num_coeffs * (1 - compression_ratio));
    
    % Set the coefficients beyond the threshold to zero
    dct_coeffs(sorted_indices(threshold_index+1:end)) = 0;
    
    % Reconstruct the DCT matrix from the modified coefficients
    dct_image_compressed = reshape(dct_coeffs, size(dct_image));
end
