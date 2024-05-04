function dct_image = my_dct2_block(image, block_size)
    % Get the size of the image
    [rows, cols] = size(image);

    % Initialize the DCT matrix
    dct_matrix = zeros(rows, cols);

    % Compute DCT for each block
    for i = 0:(rows/block_size - 1)
        for j = 0:(cols/block_size - 1)
            % Extract block
            block = image(block_size*i+1:block_size*(i+1), block_size*j+1:block_size*(j+1));

            % Compute 2D DCT for the block
            dct_block = compute_dct(block, block_size);

            % Store DCT coefficients in the output matrix
            dct_matrix(block_size*i+1:block_size*(i+1), block_size*j+1:block_size*(j+1)) = dct_block;
        end
    end

    % Return the DCT image
    dct_image = dct_matrix;
end

function dct_block = compute_dct(block, block_size)
    % Get the size of the block
    [rows, cols] = size(block);

    % Initialize DCT block
    dct_block = zeros(rows, cols);

    % Compute DCT
    for m = 0:rows-1
        for n = 0:cols-1
            sum_val = 0;
            for x = 0:rows-1
                for y = 0:cols-1
                    sum_val = sum_val + block(x+1, y+1) * ...
                        cos(pi*(2*x+1)*m/(2*rows)) * ...
                        cos(pi*(2*y+1)*n/(2*cols));
                end
            end
            if m == 0
                alpha_m = 1 / sqrt(rows);
            else
                alpha_m = sqrt(2) / sqrt(rows);
            end
            if n == 0
                alpha_n = 1 / sqrt(cols);
            else
                alpha_n = sqrt(2) / sqrt(cols);
            end
            dct_block(m+1, n+1) = alpha_m * alpha_n * sum_val;
        end
    end
end
