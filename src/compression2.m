function dct_image_compressed = compression(dct_image, block_size, compression_ratio)
    % Get the size of the DCT image
    [rows, cols] = size(dct_image);
    
    % Calculate the number of coefficients to keep per block
    num_coeffs_to_keep = round((1-compression_ratio) * block_size);
    
    % Iterate over blocks and compress each block separately
    for i = 1:block_size:rows
        for j = 1:block_size:cols
            % Determine the boundaries of the current block
            row_end = min(i+block_size-1, rows);
            col_end = min(j+block_size-1, cols);
            
            % Get the current block
            current_block = dct_image(i:row_end, j:col_end);
            
            % Zero out coefficients beyond the compression ratio along rows and columns
            current_block(num_coeffs_to_keep+1:end, :) = 0;
            current_block(:, num_coeffs_to_keep+1:end) = 0;
            
            % Update the compressed block in the DCT image
            dct_image(i:row_end, j:col_end) = current_block;
        end
    end
    
    % Return the compressed DCT image
    dct_image_compressed = dct_image;
end
