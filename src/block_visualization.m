function block_visualization(image, block_size)
    % Get dimensions of the image
    [rows, cols] = size(image);
    
    % Calculate the number of blocks in each dimension
    num_blocks_row = floor(rows / block_size);
    num_blocks_col = floor(cols / block_size);
    
    % Initialize a cell array to store non-empty blocks
    blocks = cell(num_blocks_row * num_blocks_col, 1);
    
    % Extract non-empty blocks from the image
    idx = 1;
    for i = 1:num_blocks_row
        for j = 1:num_blocks_col
            % Calculate the starting and ending row and column indices for the current block
            start_row = (i - 1) * block_size + 1;
            end_row = min(start_row + block_size - 1, rows);
            start_col = (j - 1) * block_size + 1;
            end_col = min(start_col + block_size - 1, cols);
            
            % Extract the block from the image
            block = image(start_row:end_row, start_col:end_col);
            
            % Check if the block is not empty (contains pixels other than white)
            if any(block(:) ~= 255) % Assuming white is represented by 255
                % Store the block in the cell array
                blocks{idx} = block;
                idx = idx + 1;
            end
        end
    end
    
    % Trim empty cells from the blocks array
    blocks(idx:end) = [];
    
    % Create a montage of the non-empty blocks
    figure;
    montage(blocks, 'Size', [num_blocks_row, num_blocks_col], 'BorderSize',[2,2], 'BackgroundColor', 'white');
end
