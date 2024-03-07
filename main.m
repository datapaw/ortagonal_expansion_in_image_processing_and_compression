%% Ideal sampling
%importing the image
Image = imread('cat.jpg');
imshow(Image)

%downsampling by how many times
DS = 3;

%obtaining the size of the image using:
%size(rows/column/page)
%Row - number of rows in the image / height
%Column - number of columns / width
%Page - encodes the three color channels RGB
[Height, Width, RGB] = size(Image);

%sampling calculations
%DS - downsampling
DSimage_1 = Image(1:DS:Height, 1:DS:Width);
figure; imshow(DSimage_1); title(["Downsampled by" num2str(DS)])

%downsampling and resizing up
%RS - resizing up
RSimage_1 = imresize(DSimage_1,[Height,Width]);
figure;imshow(RSimage_1); title("Downsampling and resizing up")

%downsampling and resizing up with filter
%RSF - resizing up + filter
RSFimage_1 = imfilter(RSimage_1, fspecial("laplacian"));
figure
subplot(2,1,1);imshow(RSFimage_1); title("Downsampling and resizing up with laplacian filter")

RSFimage_2 = imfilter(RSimage_1, fspecial("gaussian",6,2));
subplot(2,1,2);imshow(RSFimage_2); title("Downsampling and resizing up with gaussian filter")

%% Non-ideal sampling








