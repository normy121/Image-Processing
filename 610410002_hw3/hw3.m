clc;
close all;
clear;

rgbImage = imread('./HW3_test_image/kitchen.jpg');
img = double(rgbImage)/255;
img_RGB = img;
R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);
img_RGB(:,:,1) = double(my_histogram(uint8(img_RGB(:,:,1)*255)));
img_RGB(:,:,2) = double(my_histogram(uint8(img_RGB(:,:,2)*255)));
img_RGB(:,:,3) = double(my_histogram(uint8(img_RGB(:,:,3)*255)));

%RGB to HSI
H = acosd(((1/2)*((R-G)+(R-B)))./((((R-G).^2+((R-B).*(G-B))).^0.5)+eps));
if B > G
    H = 360 - acosd((((R-G)+(R-B))./2)./(sqrt((R-G).^2+(R-B).*(G-B))));
end
H = H/360;  %Normalized
S= 1 - (3./(R+G+B+0.000001)).*min(img,[],3);
I = (R+G+B)./3;
%My histeq
HSI = zeros(size(rgbImage));
HSI(:,:,1) = H;
HSI(:,:,2) = S;
HSI(:,:,3) = double(my_histogram(uint8(I*255)));
%HSI to RGB
H1=HSI(:,:,1);  
S1=HSI(:,:,2);  
I1=HSI(:,:,3);  
    
R1=zeros(size(H1));  
G1=zeros(size(H1));  
B1=zeros(size(H1));  
img_HSI=zeros([size(H1),3]);  
%Multiply Hue by 360 to represent in the range [0 360]  
H1=H1*360;                                               
%RG Sector(0<=H<120)  
%When H is in the above sector, the RGB components equations are  
B1(H1<120)=I1(H1<120).*(1-S1(H1<120));
R1(H1<120)=I1(H1<120).*(1+((S1(H1<120).*cosd(H1(H1<120)))./cosd(60-H1(H1<120))));  
G1(H1<120)=3.*I1(H1<120)-(R1(H1<120)+B1(H1<120));  
%GB Sector(120<=H<240)  
%When H is in the above sector, the RGB components equations are  
%Subtract 120 from Hue  
H2=H1-120;  
R1(H1>=120&H1<240)=I1(H1>=120&H1<240).*(1-S1(H1>=120&H1<240));  
G1(H1>=120&H1<240)=I1(H1>=120&H1<240).*(1+((S1(H1>=120&H1<240).*cosd(H2(H1>=120&H1<240)))./cosd(60-H2(H1>=120&H1<240))));  
B1(H1>=120&H1<240)=3.*I1(H1>=120&H1<240)-(R1(H1>=120&H1<240)+G1(H1>=120&H1<240));  
%BR Sector(240<=H<=360)  
%When H is in the above sector, the RGB components equations are  
%Subtract 240 from Hue  
H2=H1-240;  
G1(H1>=240&H1<=360)=I1(H1>=240&H1<=360).*(1-S1(H1>=240&H1<=360));  
B1(H1>=240&H1<=360)=I1(H1>=240&H1<=360).*(1+((S1(H1>=240&H1<=360).*cosd(H2(H1>=240&H1<=360)))./cosd(60-H2(H1>=240&H1<=360))));  
R1(H1>=240&H1<=360)=3.*I1(H1>=240&H1<=360)-(G1(H1>=240&H1<=360)+B1(H1>=240&H1<=360));  
%Form RGB Image  
img_HSI(:,:,1)=R1;  
img_HSI(:,:,2)=G1;  
img_HSI(:,:,3)=B1;  
%Represent the image in the range [0 255]  

[L1,a1,b1] = my_RGB2Lab(R,G,B);
new_Lab = double(my_histogram(uint8(L1)));
[R1,G1,B1] = my_Lab2RGB(new_Lab*100,a1,b1);
img_Lab = cat(3,R1,G1,B1);

figure(1),
set(figure(1), 'WindowState', 'maximized');
subplot(4,4,1);
imshow(rgbImage);title('Origin kitchen');
subplot(4,4,2);
imshow(img_RGB);title('RGB kitchen');
subplot(4,4,3);
imshow(img_HSI);title('HSI kitchen');
subplot(4,4,4);
imshow(img_Lab);title('Lab kitchen');

%========================================================================%
rgbImage = imread('./HW3_test_image/house.jpg');
img = double(rgbImage)/255;
img_RGB = img;
R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);
img_RGB(:,:,1) = double(my_histogram(uint8(img_RGB(:,:,1)*255)));
img_RGB(:,:,2) = double(my_histogram(uint8(img_RGB(:,:,2)*255)));
img_RGB(:,:,3) = double(my_histogram(uint8(img_RGB(:,:,3)*255)));

%RGB to HSI
H = acosd(((1/2)*((R-G)+(R-B)))./((((R-G).^2+((R-B).*(G-B))).^0.5)+eps));
if B > G
    H = 360 - acosd((((R-G)+(R-B))./2)./(sqrt((R-G).^2+(R-B).*(G-B))));
end
H = H/360;  %Normalized
S= 1 - (3./(R+G+B+0.000001)).*min(img,[],3);
I = (R+G+B)./3;
%My histeq
HSI = zeros(size(rgbImage));
HSI(:,:,1) = H;
HSI(:,:,2) = S;
HSI(:,:,3) = double(my_histogram(uint8(I*255)));
%HSI to RGB
H1=HSI(:,:,1);  
S1=HSI(:,:,2);  
I1=HSI(:,:,3);  
    
R1=zeros(size(H1));  
G1=zeros(size(H1));  
B1=zeros(size(H1));  
img_HSI=zeros([size(H1),3]);  
%Multiply Hue by 360 to represent in the range [0 360]  
H1=H1*360;                                               
%RG Sector(0<=H<120)  
%When H is in the above sector, the RGB components equations are  
B1(H1<120)=I1(H1<120).*(1-S1(H1<120));
R1(H1<120)=I1(H1<120).*(1+((S1(H1<120).*cosd(H1(H1<120)))./cosd(60-H1(H1<120))));  
G1(H1<120)=3.*I1(H1<120)-(R1(H1<120)+B1(H1<120));  
%GB Sector(120<=H<240)  
%When H is in the above sector, the RGB components equations are  
%Subtract 120 from Hue  
H2=H1-120;  
R1(H1>=120&H1<240)=I1(H1>=120&H1<240).*(1-S1(H1>=120&H1<240));  
G1(H1>=120&H1<240)=I1(H1>=120&H1<240).*(1+((S1(H1>=120&H1<240).*cosd(H2(H1>=120&H1<240)))./cosd(60-H2(H1>=120&H1<240))));  
B1(H1>=120&H1<240)=3.*I1(H1>=120&H1<240)-(R1(H1>=120&H1<240)+G1(H1>=120&H1<240));  
%BR Sector(240<=H<=360)  
%When H is in the above sector, the RGB components equations are  
%Subtract 240 from Hue  
H2=H1-240;  
G1(H1>=240&H1<=360)=I1(H1>=240&H1<=360).*(1-S1(H1>=240&H1<=360));  
B1(H1>=240&H1<=360)=I1(H1>=240&H1<=360).*(1+((S1(H1>=240&H1<=360).*cosd(H2(H1>=240&H1<=360)))./cosd(60-H2(H1>=240&H1<=360))));  
R1(H1>=240&H1<=360)=3.*I1(H1>=240&H1<=360)-(G1(H1>=240&H1<=360)+B1(H1>=240&H1<=360));  
%Form RGB Image  
img_HSI(:,:,1)=R1;  
img_HSI(:,:,2)=G1;  
img_HSI(:,:,3)=B1;  
%Represent the image in the range [0 255]  

[L1,a1,b1] = my_RGB2Lab(R,G,B);
new_Lab = double(my_histogram(uint8(L1)));
[R1,G1,B1] = my_Lab2RGB(new_Lab*100,a1,b1);
img_Lab = cat(3,R1,G1,B1);

subplot(4,4,5);
imshow(rgbImage);title('Origin house');
subplot(4,4,6);
imshow(img_RGB);title('RGB house');
subplot(4,4,7);
imshow(img_HSI);title('HSI house');
subplot(4,4,8);
imshow(img_Lab);title('Lab house');

%========================================================================%
rgbImage = imread('./HW3_test_image/church.jpg');
img = double(rgbImage)/255;
img_RGB = img;
R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);
img_RGB(:,:,1) = double(my_histogram(uint8(img_RGB(:,:,1)*255)));
img_RGB(:,:,2) = double(my_histogram(uint8(img_RGB(:,:,2)*255)));
img_RGB(:,:,3) = double(my_histogram(uint8(img_RGB(:,:,3)*255)));

%RGB to HSI
H = acosd(((1/2)*((R-G)+(R-B)))./((((R-G).^2+((R-B).*(G-B))).^0.5)+eps));
if B > G
    H = 360 - acosd((((R-G)+(R-B))./2)./(sqrt((R-G).^2+(R-B).*(G-B))));
end
H = H/360;  %Normalized
S= 1 - (3./(R+G+B+0.000001)).*min(img,[],3);
I = (R+G+B)./3;
%My histeq
HSI = zeros(size(rgbImage));
HSI(:,:,1) = H;
HSI(:,:,2) = S;
HSI(:,:,3) = double(my_histogram(uint8(I*255)));
%HSI to RGB
H1=HSI(:,:,1);  
S1=HSI(:,:,2);  
I1=HSI(:,:,3);  
    
R1=zeros(size(H1));  
G1=zeros(size(H1));  
B1=zeros(size(H1));  
img_HSI=zeros([size(H1),3]);  
%Multiply Hue by 360 to represent in the range [0 360]  
H1=H1*360;                                               
%RG Sector(0<=H<120)  
%When H is in the above sector, the RGB components equations are  
B1(H1<120)=I1(H1<120).*(1-S1(H1<120));
R1(H1<120)=I1(H1<120).*(1+((S1(H1<120).*cosd(H1(H1<120)))./cosd(60-H1(H1<120))));  
G1(H1<120)=3.*I1(H1<120)-(R1(H1<120)+B1(H1<120));  
%GB Sector(120<=H<240)  
%When H is in the above sector, the RGB components equations are  
%Subtract 120 from Hue  
H2=H1-120;  
R1(H1>=120&H1<240)=I1(H1>=120&H1<240).*(1-S1(H1>=120&H1<240));  
G1(H1>=120&H1<240)=I1(H1>=120&H1<240).*(1+((S1(H1>=120&H1<240).*cosd(H2(H1>=120&H1<240)))./cosd(60-H2(H1>=120&H1<240))));  
B1(H1>=120&H1<240)=3.*I1(H1>=120&H1<240)-(R1(H1>=120&H1<240)+G1(H1>=120&H1<240));  
%BR Sector(240<=H<=360)  
%When H is in the above sector, the RGB components equations are  
%Subtract 240 from Hue  
H2=H1-240;  
G1(H1>=240&H1<=360)=I1(H1>=240&H1<=360).*(1-S1(H1>=240&H1<=360));  
B1(H1>=240&H1<=360)=I1(H1>=240&H1<=360).*(1+((S1(H1>=240&H1<=360).*cosd(H2(H1>=240&H1<=360)))./cosd(60-H2(H1>=240&H1<=360))));  
R1(H1>=240&H1<=360)=3.*I1(H1>=240&H1<=360)-(G1(H1>=240&H1<=360)+B1(H1>=240&H1<=360));  
%Form RGB Image  
img_HSI(:,:,1)=R1;  
img_HSI(:,:,2)=G1;  
img_HSI(:,:,3)=B1;  
%Represent the image in the range [0 255]  

[L1,a1,b1] = my_RGB2Lab(R,G,B);
new_Lab = double(my_histogram(uint8(L1)));
[R1,G1,B1] = my_Lab2RGB(new_Lab*100,a1,b1);
img_Lab = cat(3,R1,G1,B1);

subplot(4,4,9);
imshow(rgbImage);title('Origin church');
subplot(4,4,10);
imshow(img_RGB);title('RGB church');
subplot(4,4,11);
imshow(img_HSI);title('HSI church');
subplot(4,4,12);
imshow(img_Lab);title('Lab church');

%========================================================================%
rgbImage = imread('./HW3_test_image/aloe.jpg');
img = double(rgbImage)/255;
img_RGB = img;
R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);
img_RGB(:,:,1) = double(my_histogram(uint8(img_RGB(:,:,1)*255)));
img_RGB(:,:,2) = double(my_histogram(uint8(img_RGB(:,:,2)*255)));
img_RGB(:,:,3) = double(my_histogram(uint8(img_RGB(:,:,3)*255)));

%RGB to HSI
H = acosd(((1/2)*((R-G)+(R-B)))./((((R-G).^2+((R-B).*(G-B))).^0.5)+eps));
if B > G
    H = 360 - acosd((((R-G)+(R-B))./2)./(sqrt((R-G).^2+(R-B).*(G-B))));
end
H = H/360;  %Normalized
S= 1 - (3./(R+G+B+0.000001)).*min(img,[],3);
I = (R+G+B)./3;
%My histeq
HSI = zeros(size(rgbImage));
HSI(:,:,1) = H;
HSI(:,:,2) = S;
HSI(:,:,3) = double(my_histogram(uint8(I*255)));
%HSI to RGB
H1=HSI(:,:,1);  
S1=HSI(:,:,2);  
I1=HSI(:,:,3);  
    
R1=zeros(size(H1));  
G1=zeros(size(H1));  
B1=zeros(size(H1));  
img_HSI=zeros([size(H1),3]);  
%Multiply Hue by 360 to represent in the range [0 360]  
H1=H1*360;                                               
%RG Sector(0<=H<120)  
%When H is in the above sector, the RGB components equations are  
B1(H1<120)=I1(H1<120).*(1-S1(H1<120));
R1(H1<120)=I1(H1<120).*(1+((S1(H1<120).*cosd(H1(H1<120)))./cosd(60-H1(H1<120))));  
G1(H1<120)=3.*I1(H1<120)-(R1(H1<120)+B1(H1<120));  
%GB Sector(120<=H<240)  
%When H is in the above sector, the RGB components equations are  
%Subtract 120 from Hue  
H2=H1-120;  
R1(H1>=120&H1<240)=I1(H1>=120&H1<240).*(1-S1(H1>=120&H1<240));  
G1(H1>=120&H1<240)=I1(H1>=120&H1<240).*(1+((S1(H1>=120&H1<240).*cosd(H2(H1>=120&H1<240)))./cosd(60-H2(H1>=120&H1<240))));  
B1(H1>=120&H1<240)=3.*I1(H1>=120&H1<240)-(R1(H1>=120&H1<240)+G1(H1>=120&H1<240));  
%BR Sector(240<=H<=360)  
%When H is in the above sector, the RGB components equations are  
%Subtract 240 from Hue  
H2=H1-240;  
G1(H1>=240&H1<=360)=I1(H1>=240&H1<=360).*(1-S1(H1>=240&H1<=360));  
B1(H1>=240&H1<=360)=I1(H1>=240&H1<=360).*(1+((S1(H1>=240&H1<=360).*cosd(H2(H1>=240&H1<=360)))./cosd(60-H2(H1>=240&H1<=360))));  
R1(H1>=240&H1<=360)=3.*I1(H1>=240&H1<=360)-(G1(H1>=240&H1<=360)+B1(H1>=240&H1<=360));  
%Form RGB Image  
img_HSI(:,:,1)=R1;  
img_HSI(:,:,2)=G1;  
img_HSI(:,:,3)=B1;  
%Represent the image in the range [0 255]  

[L1,a1,b1] = my_RGB2Lab(R,G,B);
new_Lab = double(my_histogram(uint8(L1)));
[R1,G1,B1] = my_Lab2RGB(new_Lab*100,a1,b1);
img_Lab = cat(3,R1,G1,B1);

subplot(4,4,13);
imshow(rgbImage);title('Origin aloe');
subplot(4,4,14);
imshow(img_RGB);title('RGB aloe');
subplot(4,4,15);
imshow(img_HSI);title('HSI aloe');
subplot(4,4,16);
imshow(img_Lab);title('Lab aloe');

function new_img = my_histogram(part_img)
    [row,col] = size(part_img);
    all_pixel = row * col;
    %Probaility Density Function
    pixel_count = zeros(1,256);
    for i=1:row 
        for j=1:col
            pixel_count(part_img(i,j) + 1) = pixel_count(part_img(i,j) + 1) + 1;
        end
    end
    part_img_pdf = pixel_count / all_pixel;

    %Cumulative Distribution Function 累加
    part_img_cdf = zeros(1,256);
    part_img_cdf(1) = part_img_pdf(1);
    for i=2:256
        part_img_cdf(i) = part_img_cdf(i-1) + part_img_pdf(i);
    end
    
    %Make new image
    new_img = zeros(row,col);
    for i=1:row
        for j=1:col
            new_img(i,j) = part_img_cdf(part_img(i,j) + 1);
        end
    end
end

function [L,a,b] = my_RGB2Lab(R,G,B)
    if nargin == 1
        B = double(R(:,:,3));
        G = double(R(:,:,2));
        R = double(R(:,:,1));
    end
    if max(max(R)) > 1.0 || max(max(G)) > 1.0 || max(max(B)) > 1.0
        R = double(R) / 255;
        G = double(G) / 255;
        B = double(B) / 255;
    end
    % Set a threshold
    T = 0.008856;
    [M, N] = size(R);
    s = M * N;
    RGB = [reshape(R,1,s); reshape(G,1,s); reshape(B,1,s)];
    % RGB to XYZ
    MAT = [0.412453 0.357580 0.180423;
           0.212671 0.715160 0.072169;
           0.019334 0.119193 0.950227];
    XYZ = MAT * RGB;
    X = XYZ(1,:) / 0.950456;
    Y = XYZ(2,:);
    Z = XYZ(3,:) / 1.088754;
    XT = X > T;
    YT = Y > T;
    ZT = Z > T;
    Y3 = Y.^(1/3); 
    fX = XT .* X.^(1/3) + (~XT) .* (7.787 .* X + 16/116);
    fY = YT .* Y3 + (~YT) .* (7.787 .* Y + 16/116);
    fZ = ZT .* Z.^(1/3) + (~ZT) .* (7.787 .* Z + 16/116);
    L = reshape(YT .* (116 * Y3 - 16.0) + (~YT) .* (903.3 * Y), M, N);
    a = reshape(500 * (fX - fY), M, N);
    b = reshape(200 * (fY - fZ), M, N);
    if nargout < 2
        L = cat(3,L,a,b);
    end
end

function [R, G, B] = my_Lab2RGB(L, a, b)
    if nargin == 1
        b = L(:,:,3);
        a = L(:,:,2);
        L = L(:,:,1);
    end
    % Thresholds
    T1 = 0.008856;
    T2 = 0.206893;
    [M, N] = size(L);
    s = M * N;
    L = reshape(L, 1, s);
    a = reshape(a, 1, s);
    b = reshape(b, 1, s);
    % Compute Y
    fY = ((L + 16) / 116) .^ 3;
    YT = fY > T1;
    fY = (~YT) .* (L / 903.3) + YT .* fY;
    Y = fY;
    % Alter fY slightly for further calculations
    fY = YT .* (fY .^ (1/3)) + (~YT) .* (7.787 .* fY + 16/116);
    % Compute X
    fX = a / 500 + fY;
    XT = fX > T2;
    X = (XT .* (fX .^ 3) + (~XT) .* ((fX - 16/116) / 7.787));
    % Compute Z
    fZ = fY - b / 200;
    ZT = fZ > T2;
    Z = (ZT .* (fZ .^ 3) + (~ZT) .* ((fZ - 16/116) / 7.787));
    % Normalize for D65 white point
    X = X * 0.950456;
    Z = Z * 1.088754;
    % XYZ to RGB
    MAT = [ 3.240479 -1.537150 -0.498535;
           -0.969256  1.875992  0.041556;
            0.055648 -0.204043  1.057311];
    RGB = max(min(MAT * [X; Y; Z], 1), 0);
    R = reshape(RGB(1,:), M, N);
    G = reshape(RGB(2,:), M, N);
    B = reshape(RGB(3,:), M, N); 
    if nargout < 2
        R = uint8(round(cat(3,R,G,B) * 255));
    end
end

