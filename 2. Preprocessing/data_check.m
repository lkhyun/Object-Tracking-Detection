clc; clear; close all;
base_dir = 'C:\학부연구생\rader\ex_dataset\[2401~2402] UWB_Biopac_Bed dataset\';
sub_n = 1
file_name = sprintf('label_data_subject_%d.mat', sub_n);
file_path = fullfile(base_dir, sprintf('%d\\', sub_n), file_name);

load (file_path)

%%
clc; clear; close all;
base_dir = 'C:\학부연구생\pvdf\';
cd(base_dir)
sub_n = 22
file_name = sprintf('label_data_7500_96_22_%d.mat', sub_n);
file_path = fullfile(base_dir, sprintf('%d\\', sub_n), file_name);

load("label_data_7500_96_22.mat")
%%
data = label_data{2};
data_1col = data(:,:,1);