
%% UWB raw data extraction from .dat file

clear,clc,close all;

cd('C:\Users\dlrkd\Desktop\hai-lab\UWB_biopac_sync_code\[Lib] x64_MSVCR141');
if not(libisloaded('SmartBRMonitoringLib'))
    loadlibrary('SmartBRMonitoringLib.dll', 'SBRMEstimator.h', 'addheader', 'SBRMITypes.h');
end
libfunctions('SmartBRMonitoringLib');

Fs_uwb = 17;
%%%%%%%%%%%%%%%%%%%%%%% UWB raw data extraction %%%%%%%%%%%%%%%%%%%%%%%%%%%

dir_name = ['C:\Users\dlrkd\Desktop\hai-lab\testreal'];

current_dir = dir([sprintf('%s', dir_name)]);
idx1 = find(~cellfun(@isempty, strfind({current_dir.name}, 'com')));
idx2 = find(~cellfun(@isempty, strfind({current_dir.name}, 'tv')));
fname1 = current_dir(idx1).name
fname2 = current_dir(idx2).name

% 반복 실행 횟수
num_iterations = 2;

% 경로 설정
radarpath = {fname1, fname2};

for iter = 1:num_iterations
    current_radarpath = radarpath{iter};

    cd(sprintf('%s\\%s', dir_name, current_radarpath))

    FileList = dir('xethru_datafloat_*.dat');
    FileName = FileList.name;
    %시작시간 추출
    split_FileName = strsplit(FileName, '_');
    split_FileName_dot = strsplit(split_FileName{4},'.');
    split_time = cellstr(reshape(split_FileName_dot{1}, 2, [])');
    numbers = str2double(split_time);
    start_time = 3600*numbers(1) + 60*numbers(2) + numbers(3);

    fid_raw = fopen(FileName,'r');

    pOnePacketSize = libpointer('int32Ptr', zeros(1,1));
    calllib('SmartBRMonitoringLib','SBRMEstimator_Initialize', Fs_uwb, pOnePacketSize);

    sample_count = 0;
    sample_drop_period = 434;
    DataCursor = 1;
    rawdata_computer = [];
    InputData = [];
    while (1)

        id = fread(fid_raw,1,'uint32');
        if length(id) < 1, break; end
        loop_cnt = fread(fid_raw,1,'uint32');
        numCountersFromFile = fread(fid_raw,1,'uint32');
        Data = fread(fid_raw,[1 numCountersFromFile],'real*4');

        sample_count = sample_count + 1;
        if mod(sample_count, sample_drop_period) == 0
            continue;
        end

        fInputData = single(Data');
        InputData(:,DataCursor)= double(fInputData); % Raw data
        DataCursor = DataCursor + 1;

    end
    rawdata_computer = [rawdata_computer InputData];
    fid_raw = fclose(fid_raw);

    cd(['C:\Users\dlrkd\Desktop\hai-lab\testreal\']);
    if iter == 2
        UWB_rawdata_tv = rawdata_computer;
        start_time_tv = start_time;
        save UWB_rawdata_tv.mat UWB_rawdata_tv
        break;
    end
    UWB_rawdata_computer = rawdata_computer;
    start_time_computer = start_time;
    save UWB_rawdata_computer.mat UWB_rawdata_computer
end
clearvars -except start_time_computer start_time_tv

load UWB_rawdata_computer.mat
load UWB_rawdata_tv.mat

% 두 레이더의 시작 시간을 비교하여 조정
if start_time_tv > start_time_computer
    start_time_UWB = start_time_tv;
    start_index = round((start_time_tv - start_time_computer) * 17);  % 17Hz sampling rate
    %UWB_rawdata_computer = UWB_rawdata_computer(:, start_index+1:end);
elseif start_time_tv < start_time_computer
    start_time_UWB = start_time_computer;
    start_index = round((start_time_computer - start_time_tv) * 17);  % 17Hz sampling rate
    %UWB_rawdata_tv = UWB_rawdata_tv(:, start_index+1:end);
else
    start_time_UWB = start_time_tv;
end

% 데이터 결합
%min_length = min(size(UWB_rawdata_computer, 2), size(UWB_rawdata_tv, 2));
%UWB_synchronized_data = struct('UWB_rawdata_computer',UWB_rawdata_computer(:, 1:min_length),'UWB_rawdata_tv',UWB_rawdata_tv(:, 1:min_length));
UWB_synchronized_data = struct('UWB_rawdata_computer',UWB_rawdata_computer,'UWB_rawdata_tv',UWB_rawdata_tv);

save UWB_synchronized_data.mat UWB_synchronized_data




% %% Biopac & UWB plot`
% 
% load UWB_synchronized_data.mat
% FileList = dir('*_BIOPAC_*.mat');
% FileName = FileList.name;
% load(FileName);
% 
% Fs_biopac = 250;
% Biopac_resp = data(:,1);
% 
% Fs_uwb = 17;
% UWB_rawdata = UWB_rawdata_tv;
% UWB_Idx = 150;
% 
% UWB_resp_test = UWB_rawdata(UWB_Idx,:);
% 
% 
% biopac_time = [1:length(Biopac_resp)];
% biopac_time = biopac_time/Fs_biopac;
% 
% uwb_time = [1:length(UWB_resp_test)];
% uwb_time = uwb_time/Fs_uwb;
% 
% 
% fig = figure;
% fig.WindowState = 'maximized';
% subplot(311); plot(uwb_time, UWB_resp_test, 'b'); axis tight; title('UWB respiration'); xlim([1 500]);
% subplot(312); plot(biopac_time, Biopac_resp, 'r'); axis tight; title('Biopac respiration'); xlim([1 500]);
% 
% 
% % Biopac & UWB sync
% 
% clearvars -except start_time_UWB UWB_synchronized_data FileName data
% 
% split_FileName = strsplit(FileName, '_');
% hour = strsplit(split_FileName{6},'T');
% minute = split_FileName{7}
% second = strsplit(split_FileName{8},'.');
% hour = str2double(hour{2});
% minute = str2double(minute);
% second = str2double(second{1});
% start_time_biopac = 3600*hour + 60*minute + second;
% noise_time = 5; %첫번째 노이즈까지의 시간 플롯 확인 후 판단
% 
% %두 레이더의 시작 시간을 비교하여 조정
% if start_time_UWB > start_time_biopac
%     start_index_UWB = round(noise_time * 17);  % 17Hz sampling rate
%     start_index_biopac = round((start_time_UWB - start_time_biopac + noise_time) * 250);  % 250Hz sampling rate
% elseif start_time_UWB < start_time_biopac  
%     start_index_UWB = round((start_time_biopac - start_time_UWB + noise_time) * 17);  % 17Hz sampling rate
%     start_index_biopac = round(noise_time * 250);  % 250Hz sampling rate
% else
%     start_index_UWB = round(noise_time * 17);
%     start_index_biopac = round(noise_time * 250);
% end
% UWB_synchronized_data.UWB_rawdata_computer = UWB_synchronized_data.UWB_rawdata_computer(:, start_index_UWB+1:end);
% UWB_synchronized_data.UWB_rawdata_tv = UWB_synchronized_data.UWB_rawdata_tv(:, start_index_UWB+1:end);
% data = data(start_index_biopac+1:end,:);
% 
% %데이터 결합
% UWB_synchronized_data = setfield(UWB_synchronized_data,'biopac',data);
% 
% save UWB_synchronized_data.mat UWB_synchronized_data

%% Remove Clutter
clearvars -except UWB_synchronized_data
%%%%%%%%%%%%%%%%%%%% UWB parameters extraction %%%%%%%%%%%%%%%%%%%%%%%%
Fs_uwb = 17;
rawdata_computer = UWB_synchronized_data.UWB_rawdata_computer(90:449, :);
rawdata_tv = UWB_synchronized_data.UWB_rawdata_tv(90:449, :); 
%rawdata_computer = UWB_synchronized_data.UWB_rawdata_computer(1:560, :);
%rawdata_tv = UWB_synchronized_data.UWB_rawdata_tv(1:560, :); 

numCounters = size(rawdata_computer,1);
interval = 50;

length_cnt = length(rawdata_computer(1,:));
obj = UWBProcessingVital(Fs_uwb, numCounters , 'x4');
uwb_resp = [];
uwb_hr = [];
state = [];
filtered_sig = [];
baseband_sig = [];
doppler_map = [];
br_idx = [];
DataCursor = 1;

while (1)
    if obj.frameCNT > length_cnt
        break;
    else
        sprintf('%d / %d', obj.frameCNT, length_cnt)
        RawData = rawdata_computer(1:numCounters,obj.frameCNT)';
        obj = process(obj, RawData, interval);
        uwb_resp = [uwb_resp [obj.RespSig(end); obj.estimated_BR; obj.brIdx]];
        uwb_hr = [uwb_hr [obj.HRSig(end); obj.estimated_HR]];
        state = [state obj.curState];
        filtered_sig = [filtered_sig obj.m_ProcessingData(end,:)']; % baseband - clutter
        baseband_sig = [baseband_sig obj.baseBand'];
        doppler_map = [doppler_map obj.m_DopplerMap(:,end)];
        br_idx = [br_idx obj.brIdx];
    end
end

UWBdata_com.rawdata = rawdata_computer;
UWBdata_com.resp = uwb_resp;
UWBdata_com.hr = uwb_hr;
UWBdata_com.state = state;
UWBdata_com.filtered_sig = COLUMN_DATA_VECTOR_NORMALIZATION(filtered_sig);
UWBdata_com.baseband_sig = COLUMN_DATA_VECTOR_NORMALIZATION(baseband_sig);
UWBdata_com.doppler_map = doppler_map;
UWBdata_com.br_idx = br_idx;
UWBdata_com.Fs_uwb = Fs_uwb;

numCounters = size(rawdata_tv,1);
interval = 50;

length_cnt = length(rawdata_tv(1,:));
obj = UWBProcessingVital(Fs_uwb, numCounters , 'x4');
uwb_resp = [];
uwb_hr = [];
state = [];
filtered_sig = [];
baseband_sig = [];
doppler_map = [];
br_idx = [];
DataCursor = 1;

while (1)
    if obj.frameCNT > length_cnt
        break;
    else
        sprintf('%d / %d', obj.frameCNT, length_cnt)
        RawData = rawdata_tv(1:numCounters,obj.frameCNT)';
        obj = process(obj, RawData, interval);
        uwb_resp = [uwb_resp [obj.RespSig(end); obj.estimated_BR; obj.brIdx]];
        uwb_hr = [uwb_hr [obj.HRSig(end); obj.estimated_HR]];
        state = [state obj.curState];
        filtered_sig = [filtered_sig obj.m_ProcessingData(end,:)']; % baseband - clutter
        baseband_sig = [baseband_sig obj.baseBand'];
        doppler_map = [doppler_map obj.m_DopplerMap(:,end)];
        br_idx = [br_idx obj.brIdx];
    end
end

UWBdata_tv.rawdata = rawdata_tv;
UWBdata_tv.resp = uwb_resp;
UWBdata_tv.hr = uwb_hr;
UWBdata_tv.state = state;
UWBdata_tv.filtered_sig = COLUMN_DATA_VECTOR_NORMALIZATION(filtered_sig);
UWBdata_tv.baseband_sig = COLUMN_DATA_VECTOR_NORMALIZATION(baseband_sig);
UWBdata_tv.doppler_map = doppler_map;
UWBdata_tv.br_idx = br_idx;
UWBdata_tv.Fs_uwb = Fs_uwb;



%% Data transformation for model training
clearvars -except UWBdata_tv UWBdata_com

plot_com = UWBdata_com.filtered_sig; %플롯할 레이더 위치
plot_tv = UWBdata_tv.filtered_sig; %플롯할 레이더 위치

% plot_com = UWBdata_com.rawdata; %플롯할 레이더 위치
% plot_tv = UWBdata_tv.rawdata; %플롯할 레이더 위치

average_com = mean(plot_com,1);
std_com = std(plot_com,0,1);
average_tv = mean(plot_tv,1);
std_tv = std(plot_tv,0,1);

threshold_com = average_com + 3*std_com;
threshold_tv = average_tv + 3*std_tv;

plot_com(plot_com <= threshold_com) = NaN;
plot_com_log = log10(plot_com);

plot_tv(plot_tv <= threshold_tv) = NaN;
plot_tv_log = log10(plot_tv);

plot_com_log(isnan(plot_com_log)) = 0;
plot_tv_log(isnan(plot_tv_log)) = 0;
plot_com_smooth = movmean(plot_com_log,51,2)
plot_tv_smooth = movmean(plot_tv_log,51,2)
%% show

figure;
imagesc(plot_com_log);
colorbar;
colormap jet;
clim([min(plot_com_log(:)), max(plot_com_log(:))]);
xlabel('Time');
ylabel('Distance');
title('moving tracking');
set(gca, 'YDir', 'normal');

figure;
imagesc(plot_tv_log);
colorbar;
colormap jet;
clim([min(plot_tv_log(:)), max(plot_tv_log(:))]);
xlabel('Time');
ylabel('Distance');
title('moving tracking');
set(gca, 'YDir', 'normal');

figure;
imagesc(plot_com_smooth);
colorbar;
colormap jet;
clim([min(plot_com_smooth(:)), max(plot_com_smooth(:))]);
xlabel('Time');
ylabel('Distance');
title('moving tracking');
set(gca, 'YDir', 'normal');

figure;
imagesc(plot_tv_smooth);
colorbar;
colormap jet;
clim([min(plot_tv_smooth(:)), max(plot_tv_smooth(:))]);
xlabel('Time');
ylabel('Distance');
title('moving tracking');
set(gca, 'YDir', 'normal');

%% mat 파일로 저장
com_startpoint = 34; % 시작 지점 설정
tv_startpoint = 23; % 시작 지점 설정

%UWBFilteredData.com = plot_com_log(:,com_startpoint:com_startpoint + 4283);
%UWBFilteredData.tv = plot_tv_log(:,tv_startpoint:tv_startpoint + 4283);
UWBFilteredData.com = plot_com_smooth(:,com_startpoint:com_startpoint + 2141);
UWBFilteredData.tv = plot_tv_smooth(:,tv_startpoint:tv_startpoint + 2141);

save UWB_Extration.mat UWBFilteredData

slices = cell(1, 28);

    for i = 1:28
        startCol = (i-1) * 51 + 1;
        endCol = startCol + 51 - 1;
        slices{i} = UWBFilteredData.com(1:360, startCol:endCol, :);
    end
    
    figure;
    for i = 1:28
        subplot(4, 7, i);
        imagesc(slices{i});
        title(sprintf('Slice %d', i));
    end

slices = cell(1, 28);

    for i = 1:28
        startCol = (i-1) * 51 + 1;
        endCol = startCol + 51 - 1;
        slices{i} = UWBFilteredData.tv(1:360, startCol:endCol, :);
    end
    
    figure;
    for i = 1:28
        subplot(4, 7, i);
        imagesc(slices{i});
        title(sprintf('Slice %d', i));
    end


