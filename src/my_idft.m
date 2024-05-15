function image = my_idft(dft_image, block_size)
    % Get the size of the DFT image
    [rows, cols] = size(dft_image);

    % Initialize the image matrix
    image = zeros(rows, cols);

    % Compute IDFT for each block
    for i = 0:(rows/block_size - 1)
        for j = 0:(cols/block_size - 1)
            % Extract block
            dft_block = dft_image(block_size*i+1:block_size*(i+1), block_size*j+1:block_size*(j+1));

            % Compute 2D IDFT for the block
            block = compute_idft(dft_block, block_size);

            % Store IDFT values in the output matrix
            image(block_size*i+1:block_size*(i+1), block_size*j+1:block_size*(j+1)) = block;
        end
    end
end

function block = compute_idft(dft_block, block_size)
    % Get the size of the DFT block
    [rows, cols] = size(dft_block);

    % Initialize IDFT block
    block = zeros(rows, cols);

    % Compute IDFT
    for x = 0:rows-1
        for y = 0:cols-1
            sum_val = 0;
            for m = 0:rows-1
                for n = 0:cols-1
                    sum_val = sum_val + dft_block(m+1, n+1) * ...
                        exp(1i * 2 * pi * ((m * x / rows) + (n * y / cols)));
                end
            end
            block(x+1, y+1) = sum_val / (rows * cols);
        end
    end
end