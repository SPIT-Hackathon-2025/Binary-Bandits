clc;
close all;
clear all;

%reads image 
im = imread("C:\Users\91750\Downloads\OCR_Test_4_crop.jpg");   

%resizes the image
im = imresize(im, [480 NaN]);

%displays original image
imshow(im),title("Original Image");
imgray=im
%converts to grayscale
%imgray = rgb2gray(im);
%imgray = uint8(imgray);
%binarizes the image
imbin= imbinarize(imgray);
%imbin = imgray > 128; % Simple thresholding





%edge detection using Sobel Algorithm
im = edge(imgray, 'sobel');

%dilates the image using diamond structure
im = imdilate(im, strel('diamond', 2));

%fills holes by making it full white color
im = imfill(im, 'holes');

%extract the solid filled parts
im = imerode(im, strel('diamond', 10));

%calculates Area,Bounding Box and Image size 
Iprops=regionprops(im,'BoundingBox','Area', 'Image');

%stores the area of the first element
area = Iprops.Area;
%counts the number of elements
count = numel(Iprops);
%variable to store max area, initial value=first element's area
maxa= area;
%stores bounding box
boundingBox = Iprops.BoundingBox;
%loop to find the bounding box with the maximum area
for i=1:count
   if maxa<Iprops(i).Area
       maxa=Iprops(i).Area;
       boundingBox=Iprops(i).BoundingBox;
   end
end    

%all above steps are to find location of the number plate
%Now crop the binarized image to get the number plate only
im = imcrop(imbin, boundingBox);

%resize number plate to 240 NaN
im = imresize(im, [240 NaN]);

%clear dust
im = imopen(im, strel('rectangle', [4 4]));

%remove some object if it's width is too long or too small than 500
im = bwareaopen(~im, 500);

%get width
 [h, w] = size(im);  

figure;
imshow(im),title("Extracted No. Plate with Isolated Character");

% Read letter
Iprops=regionprops(im,'BoundingBox','Area', 'Image');
count = numel(Iprops);

noPlate=[]; % Initializing the variable of number plate string.

hold on; % this is used to currunt plot active fro green box
for i=1:count % this fo loop is used to give the green border to extracted number plate
   ow = length(Iprops(i).Image(1,:));
   oh = length(Iprops(i).Image(:,1));
   if ow<(h/2) && oh>(h/3)
       rectangle('Position', Iprops(i).BoundingBox, 'EdgeColor', 'g', 'LineWidth', 2);
   end
end
for i=1:count
   %width of the ith  character of number plate
   ow = length(Iprops(i).Image(1,:));
   %length of the ith character of number plate
   oh = length(Iprops(i).Image(:,1));
   if ow<(h/2) && oh>(h/3)
       letter=readLetter(Iprops(i).Image); % Reading the letter corresponding the binary image 'N'.
       figure; imshow(Iprops(i).Image);
       noPlate=[noPlate letter] % Appending every subsequent character in noPlate variable.
   end
end




