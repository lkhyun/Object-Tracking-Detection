%%
clc; clear; close all;

base_dir = 'C:\학부연구생\rader\ex_dataset\[2401~2402] UWB_Biopac_Bed dataset\';
frames_per_segment = 7500; % 7500 프레임 단위로 자르기

for sub_n = 39:60
    if sub_n == 42 continue; end
    if sub_n == 57 continue; end
dir_name = [base_dir, sprintf('%d\\', sub_n)];
cd(dir_name);

% 구간 정보 파일 이름 설정
segment_info_file = sprintf('segment_info_subject_%d.mat', sub_n);

% 1. PVDF 데이터 호출 및 전처리
data = load('BrLab_Biopac_SyncData.mat');
pvdfall = data.Final_SyncData.pvdf_all;
fs = data.Final_SyncData.Fs_pvdf;
lo = 1;
hi = 5;
% [b_low, a_low] = butter(5, lo / (fs / 2), 'low'); % 0.1 Hz 저역 통과 필터
% 
% [b_high, a_high] = butter(5, hi / (fs / 2), 'high'); % 6 Hz 고역 통과 필터
% 
% for ch = 1:4
%     pvdfall{ch} = filtfilt(b_low, a_low, pvdfall{ch});
%     pvdfall{ch} = filtfilt(b_high, a_high, pvdfall{ch});
% end
[b, a] = butter(5, [lo hi] / (fs / 2), 'bandpass'); % 대역 통과 필터

for ch = 1:4
    pvdfall{ch} = filtfilt(b, a, pvdfall{ch});
end

combined_pvdf = cell2mat(pvdfall');
combined_pvdf = (combined_pvdf - mean(combined_pvdf(:))) / std(combined_pvdf(:));  % 전체 정규화

for ch = 1:4
    pvdfall{ch} = combined_pvdf((ch-1)*length(pvdfall{ch}) + 1:ch*length(pvdfall{ch}));
end

cwt_data = cell(1, 4); % 모든 채널의 CWT 데이터를 저장하기 위한 셀 배열

% 2. 전처리된 데이터를 CWT 변환 후 저장
for ch = 1:4
    data_ch = pvdfall{ch};
    tms = (0:length(data_ch)-1) / fs;
    [cfs, frq] = cwt(data_ch, fs, 'FrequencyLimits', [lo hi]);

    % CWT 데이터를 저장
    CWTData.fs = fs;
    CWTData.filtData = data_ch;
    CWTData.time = tms;
    CWTData.freq = frq;
    CWTData.Power = abs(cfs);

    cwt_data{1, ch} = CWTData; % 각 채널의 CWT 데이터를 저장
end
if sub_n==1 
    disp(frq); end
save(sprintf('cwt_data_subject_%d.mat', sub_n), 'cwt_data');

if exist(segment_info_file, 'file')
    load(segment_info_file, 'segments');
    disp('기존 구간 정보를 불러왔습니다.');
else
    % 구간 정보가 없는 경우 새로 선택
    data_ch1 = pvdfall{1};
    fig = figure('Position', [100, 100, 2400, 800], 'Name', sprintf('PVDF Data Segment Selection - Subject %d', sub_n), 'NumberTitle', 'off');

    plot(data_ch1);
    title('PVDF Channel 1');
    xlabel('Sample');
    ylabel('Amplitude');
    axis tight;

    [x, ~] = ginput(8);
    segments = sort(round(x));
    segments = [1; segments; length(data_ch1)]; % 시작과 끝을 포함하여 9개의 구간으로 나눔

    % 구간 정보를 저장
    save(segment_info_file, 'segments');
    close(fig);
end

label_data = cell(9, 1); 
discarded_data = cell(9, 1); 

for step = 1:9
    start_idx = segments(step);
    end_idx = segments(step + 1) - 1;

    if end_idx > length(pvdfall{1})
        end_idx = length(pvdfall{1});
    end

    combined_cwt_power = [];

    % 0~3 채널 데이터를 통합 저장
    for ch = 1:4
        CWTData = cwt_data{1, ch};
        time_indices = (CWTData.time >= start_idx/fs) & (CWTData.time <= end_idx/fs);
        cwt_segment_power = CWTData.Power(:, time_indices);
        combined_cwt_power = [combined_cwt_power; cwt_segment_power];
    end

    % 2, 4, 6, 8 구간에 대해서만 segment slicing 수행 및 데이터 정제
    if ismember(step, [2, 4, 6, 8])
        num_segments = floor(size(combined_cwt_power, 2) / frames_per_segment);
        segment_mean = mean(combined_cwt_power, 2);   % 전체 segment의 평균 계산
        segment_var = var(combined_cwt_power, 0, 2);  % 전체 segment의 분산 계산

        for i = 1:num_segments
            segment_slice = combined_cwt_power(:, (i-1)*frames_per_segment + 1:i*frames_per_segment);

            if size(segment_slice, 2) == frames_per_segment
                slice_mean = mean(segment_slice, 2);
                slice_var = var(segment_slice, 0, 2);

                mean_difference = abs(slice_mean - segment_mean);
                variance_difference = abs(slice_var - segment_var);

                mean_threshold = 1 * abs(segment_mean);  
                var_threshold = 1 * abs(segment_var);

                if all(mean_difference <= mean_threshold) && all(variance_difference <= var_threshold)
                    if isempty(label_data{step})
                        label_data{step} = segment_slice';  % 빈 경우에는 그냥 추가
                    else
                        label_data{step} = [label_data{step}; segment_slice']; % 정제된 데이터 저장
                    end

                    % Segment slice 데이터를 시각화 및 저장
                    figure;
                    imagesc(segment_slice);
                    title(sprintf('Segment Slice %d - Subject %d, Segment %d', i, sub_n, step));
                    xlabel('Time Frame');
                    ylabel('Frequency Bin');
                    colorbar;
                    saveas(gcf, sprintf('segment_segment_%d-%d_subject_%d.png', step, i, sub_n));
                    close;
                else
                    % 버려진 데이터 저장
                    if isempty(discarded_data{step})
                        discarded_data{step} = segment_slice'; % 빈 경우에는 그냥 추가
                    else
                        discarded_data{step} = [discarded_data{step}; segment_slice']; % 버려진 데이터 저장
                    end

                    figure;
                    imagesc(segment_slice);
                    title(sprintf('Discarded Segment Slice %d - Subject %d, Segment %d', i, sub_n, step));
                    xlabel('Time Frame');
                    ylabel('Frequency Bin');
                    colorbar;
                    saveas(gcf, sprintf('discarded_segment_%d-%d_subject_%d.png', step, i, sub_n));
                    close;
                end
            end
        end
    else
        % 2, 4, 6, 8 이외의 구간은 모든 데이터 저장
        if isempty(label_data{step})
            label_data{step} = combined_cwt_power'; % 빈 경우에는 그냥 추가
        else
            label_data{step} = [label_data{step}; combined_cwt_power']; % 모든 데이터 저장
        end
    end
end

% 최종 데이터를 파일로 저장
save(sprintf('label_data_subject_%d.mat', sub_n), 'label_data');
save(sprintf('discarded_data_subject_%d.mat', sub_n), 'discarded_data'); % 버려진 데이터 저장
end