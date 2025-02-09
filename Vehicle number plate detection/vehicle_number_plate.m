% %a=imread(uigetfile('.jpg'));
% a = imread("C:\Users\91750\Downloads\OCR_Test_3.jpeg");
% 
% a=rgb2gray(a);
% figure;imshow(a);title('car');
% [r c p]=size(a);
% b=a(r/3:r,1:c);
%  imshow(b);title('LP AREA')
% [r c p]=size(b);
% Out=zeros(r,c);
% for i=1:r
%     for j=1:c
%       if b(i,j)>150
%             Out(i,j)=1;
%         else
%             Out(i,j)=0;
%         end
%     end
% end
% % load Out;
% BW3 = bwfill(Out,'holes');
% BW3=medfilt2(BW3,[4 4]);
% BW3=medfilt2(BW3,[4 4]);
% BW3=medfilt2(BW3,[4 4]);
% BW3=medfilt2(BW3,[5 5]);
% BW3=medfilt2(BW3,[5 5]);
% figure;imshow(BW3,[]);
% BW3 = bwfill(BW3,'holes');
% [L num]=bwlabel(BW3);
% STATS=regionprops(L,'all');
% disp(num);
% % close all;
% cc=[];
% removed=0;
% for i=1:num
% dd=STATS(i).Area;
% cc(i)=dd;
% 	if (dd < 50000)
%           	L(L==i)=0;
% 			removed = removed + 1;
%             num=num-1;
%     end
% end
% [L2 num2]=bwlabel(L);
% figure,imshow(L2);
%  STATS = regionprops(L2,'All');
% if num2>2
%      for i=1:num2    
% 	aa=  STATS(i).Orientation;    
% 	if aa > 0
% 
% 	imshow(L==i);    
% 	end
%      end
% 	disp('exit');
% end
%  [r c]=size(L2);
% Out=zeros(r,c);
% k=1;
%  B=STATS.BoundingBox;
% Xmin=B(2);
% Xmax=B(2)+B(4);
% Ymin=B(1)
% Ymax=B(1)+B(3);
% LP=[];
% LP=b(Xmin+25:Xmax-20,Ymin+10:Ymax-10);
% figure,imshow(LP,[]);


clc;        % Clear command window
clear;      % Clear workspace
close all;  % Close all figures

a = imread("C:\Users\91750\Downloads\OCR_Test_3.jpeg");


% Convert image to grayscale
a = rgb2gray(a);
figure, imshow(a), title('Car Image');

% Extract lower part of the image (Assuming license plate is at bottom)
[r, c, ~] = size(a);
b = a(floor(r / 3):r, 1:c);
figure, imshow(b), title('License Plate Area');
% imwrite(b,"C:\Users\91750\Downloads\OCR_Test_4_crop.jpg");
% Convert grayscale to binary image using thresholding
Out = b > 150;
Out=~Out
figure, imshow(Out), title('Binary Image');

% Fill holes and apply median filtering
BW3 = imfill(Out, 'holes');
BW3 = medfilt2(BW3, [5 5]);

% Label connected components
[L, num] = bwlabel(BW3);
STATS = regionprops(L, 'BoundingBox', 'Area', 'Orientation');
disp(['Total Components Found: ', num2str(num)]);

% Filter out small regions (noise removal)
min_area = 50000;
for i = 1:num
    if STATS(i).Area < min_area
        L(L == i) = 0;
    end
end

% Re-label after removing small components
[L2, num2] = bwlabel(L);
figure, imshow(L2), title('Filtered Components');

% Find the best bounding box for the license plate
if num2 > 0  % Ensure at least one valid region is found
    B = round(STATS(1).BoundingBox);  % Get Bounding Box of first region
    
    % Ensure valid indices within image bounds
    Xmin = max(1, round(B(2) + 25));
    Xmax = min(size(b, 1), round(B(2) + B(4) - 20));
    Ymin = max(1, round(B(1) + 10));
    Ymax = min(size(b, 2), round(B(1) + B(3) - 10));

    % Extract the license plate region
    LP = b(Xmin:Xmax, Ymin:Ymax);
    figure, imshow(LP, []), title('Extracted License Plate');
else
    disp('No valid license plate detected.');
end

