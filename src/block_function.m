function block_visualization(image, block_size)
    % Sprawdzenie czy obraz jest kolorowy czy w skali szarości
    if size(image, 3) == 3
        % Obraz kolorowy - konwertowanie do skali szarości
        image_gray = rgb2gray(image);
    else
        % Obraz w skali szarości
        image_gray = image;
    end
    
    % Rozmiar obrazu
    [rows, cols] = size(image_gray);
    
    % Liczba bloków wzdłuż każdej osi
    num_blocks_rows = floor(rows / block_size);
    num_blocks_cols = floor(cols / block_size);
    
    % Inicjalizacja tablicy na bloki
    blocks = cell(num_blocks_rows, num_blocks_cols);
    
    % Podział obrazu na bloki
    for i = 1:num_blocks_rows
        for j = 1:num_blocks_cols
            % Określenie współrzędnych bloku
            row_start = (i - 1) * block_size + 1;
            row_end = row_start + block_size - 1;
            col_start = (j - 1) * block_size + 1;
            col_end = col_start + block_size - 1;
            
            % Wycięcie bloku z obrazu
            block = image_gray(row_start:row_end, col_start:col_end);
            
            % Zapisanie bloku w tablicy
            blocks{i, j} = block;
        end
    end
    
    % Utworzenie montażu z bloków i wyświetlenie
    montage(blocks, 'Size', [num_blocks_rows, num_blocks_cols]);
end
