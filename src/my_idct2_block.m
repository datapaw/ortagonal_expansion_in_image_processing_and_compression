function inv_image = my_idct2_block(dct_image, block_size)
    % Get the size of the DCT image
    [rows, cols] = size(dct_image);

    % Initialize the inverse DCT matrix
    inv_matrix = zeros(rows, cols);

    % Compute IDCT for each block
    for i = 0:(rows/block_size - 1)
        for j = 0:(cols/block_size - 1)
            % Extract DCT coefficients block
            dct_block = dct_image(block_size*i+1:block_size*(i+1), block_size*j+1:block_size*(j+1));

            % Compute 2D IDCT for the block
            inv_block = compute_idct(dct_block, block_size);

            % Store IDCT values in the output matrix
            inv_matrix(block_size*i+1:block_size*(i+1), block_size*j+1:block_size*(j+1)) = inv_block;
        end
    end

    % Return the inverse DCT image
    inv_image = inv_matrix;
end

function inv_block = compute_idct(dct_block, block_size)
    % Get the size of the DCT block
    [rows, cols] = size(dct_block);

    % Initialize inverse DCT block
    inv_block = zeros(rows, cols);

    % Compute IDCT
    for x = 0:rows-1
        for y = 0:cols-1
            sum_val = 0;
            for m = 0:rows-1
                for n = 0:cols-1
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
                    sum_val = sum_val + alpha_m * alpha_n * dct_block(m+1, n+1) * ...
                        cos(pi*(2*x+1)*m/(2*rows)) * ...
                        cos(pi*(2*y+1)*n/(2*cols));
                end
            end
            inv_block(x+1, y+1) = sum_val;
        end
    end
end
