clc
close all;
clear;
load("C:\Users\91750\Downloads\Vehicle number plate recognition\Vehicle number plate recognition\imgfildata.mat");
[file,path]=uigetfile({'*.jpg;*.jpeg;*.png;*.tif'},'Choose an image');
s=[path,file];
img=imread(s);
np=number_plate(img);
disp(np);