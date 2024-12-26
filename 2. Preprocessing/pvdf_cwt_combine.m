%%
clc; clear; close all;
base_dir = 'C:\학부연구생\rader\ex_dataset\[2401~2402] UWB_Biopac_Bed dataset\';
        
% 전체 배열 생성 및 제거할 값들 제외
arr = setdiff(1:60, [38, 42, 57]);
window_size = 3750;
step_size = 3750; % 윈도우가 250 * 5 씩 겹치도록 설정
for j = 1:58
label_data = cell(1, 4); % cell 배열로 초기화: {0, 1, 2, 3}
    sub_n = arr(j)
    file_name = sprintf('label_data_subject_%d.mat', sub_n);
    file_path = fullfile(base_dir, sprintf('%d\\', sub_n), file_name);
    
    data = load(file_path);
    label_data_loaded = data.label_data;

    for step = [2, 4, 6, 8]
        if ~isempty(label_data_loaded{step})
            segments = label_data_loaded{step};
            
            segment_length = size(segments, 1);
            start_indices = 1:step_size:(segment_length - window_size + 1);
            
            for  start = start_indices
                segment_slice = segments(start:(start + window_size - 1), :);
                index = step / 2;
                
                if isempty(label_data{index})
                    label_data{index} = segment_slice;
                else
                    label_data{index} = cat(3, label_data{index}, segment_slice);
                end
            end
        end
    end
output_file =  sprintf('label_data_7500_96_%d.mat', sub_n);
cd('C:\학부연구생\pvdf');
save(output_file, 'label_data', '-v7.3', '-nocompression');
end
