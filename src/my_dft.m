function dft_image = my_dft(image, block_size)
    % Get the size of the image
    [rows, cols] = size(image);

    % Initialize the DFT matrix
    dft_matrix = zeros(rows, cols);

    % Compute DFT for each block
    for i = 0:(rows/block_size - 1)
        for j = 0:(cols/block_size - 1)
            % Extract block
            block = image(block_size*i+1:block_size*(i+1), block_size*j+1:block_size*(j+1));

            % Compute 2D DFT for the block
            dft_block = compute_dft(block, block_size);

            % Store DFT coefficients in the output matrix
            dft_matrix(block_size*i+1:block_size*(i+1), block_size*j+1:block_size*(j+1)) = dft_block;
        end
    end

    % Return the DFT image
    dft_image = dft_matrix;
end

function dft_block = compute_dft(block, block_size)
    % Get the size of the block
    [rows, cols] = size(block);

    % Initialize DFT block
    dft_block = zeros(rows, cols);

    % Compute DFT
    for m = 0:rows-1
        for n = 0:cols-1
            sum_val = 0;
            for x = 0:rows-1
                for y = 0:cols-1
                    sum_val = sum_val + block(x+1, y+1) * ...
                        exp(-1i * 2 * pi * ((m * x / rows) + (n * y / cols)));
                end
            end
            dft_block(m+1, n+1) = sum_val;
        end
    end
end