clear,clc,close all;
sub_n = 1
sub_name = GetSubDirsFirstLevelOnly('C:\Users\dlrkd\Desktop\hai-lab\[2401~2402] UWB_Biopac_Bed dataset');
sub_name= natsort(sub_name);
cd(['C:\Users\dlrkd\Desktop\hai-lab\[2401~2402] UWB_Biopac_Bed dataset\', sprintf('%s',sub_name{sub_n}),'\Move_Radar']);

load UWB_synchronized_data.mat
load UWB_Extration.mat

computer = UWBdata.filtered_sig;
numrows = size(computer, 1);
maxdistance = 100;

distances = linspace(1, maxdistance, numrows);

growth_rate = 0.001;

growth_factor = 1 ./ (exp(-growth_rate * distances'));

adjusted_data = computer;

average = mean(adjusted_data,1);
std = std(adjusted_data,0,1);

threshold = average + 2.2*std;

threshold_data = adjusted_data;
threshold_data(threshold_data <= threshold) = NaN;

threshold_data_log = log10(threshold_data);

figure;
imagesc(threshold_data_log);
colorbar;
colormap jet;
clim([min(threshold_data_log(:)), max(threshold_data_log(:))]);
xlabel('Time');
ylabel('Distance');
title('moving tracking');

set(gca, 'YDir', 'normal');