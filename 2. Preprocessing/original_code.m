
%% UWB raw data extraction from .dat file

clear,clc,close all;

cd('E:\Research Code\4. Projects\[23.08~24.02] BrLab\[Lib] x64_MSVCR141');
if not(libisloaded('SmartBRMonitoringLib'))
    loadlibrary('SmartBRMonitoringLib.dll', 'SBRMEstimator.h', 'addheader', 'SBRMITypes.h');
end
libfunctions('SmartBRMonitoringLib');

sub_name = GetSubDirsFirstLevelOnly('E:\Research Data\UWB_Bed_Biopac_dataset');
sub_name= natsort(sub_name);
Fs_uwb = 17;

sub_n = 4  % sub 1~60 숫자 바꾸면서 데이터 추출, s1,2,3 bed radar 데이터 측정 이상

%%%%%%%%%%%%%%%%%%%%%%% UWB raw data extraction %%%%%%%%%%%%%%%%%%%%%%%%%%%

dir_name = ['E:\Research Data\UWB_Bed_Biopac_dataset\', sprintf('%s',sub_name{sub_n})];

current_dir = dir([sprintf('%s', dir_name), '\Bed_Radar']);
idx = find(~cellfun(@isempty, strfind({current_dir.name}, 'xethru')));
if isempty(idx)==1
    idx = find(~cellfun(@isempty, strfind({current_dir.name}, 'bed')));
end
fname = current_dir(idx).name
cd(sprintf('%s\\Bed_Radar\\%s', dir_name, fname))


FileList = dir('xethru_datafloat_*.dat');
FileName = {FileList.name}';
fid_raw = fopen(FileName{1},'r');

pOnePacketSize = libpointer('int32Ptr', zeros(1,1));
calllib('SmartBRMonitoringLib','SBRMEstimator_Initialize', Fs_uwb, pOnePacketSize);

sample_count = 0;
sample_drop_period = 434;
DataCursor = 1;
rawdata = [];
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
rawdata = [rawdata InputData];
fid_raw = fclose(fid_raw);


%%%%%%%%%%%%%%%%%%%% UWB parameters extraction %%%%%%%%%%%%%%%%%%%%%%%%

numCounters = 576; % original frame = 581;
interval = 50;

length_cnt = length(rawdata(1,:));
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
        RawData = rawdata(1:numCounters,obj.frameCNT)';
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

UWBdata.rawdata = rawdata;
UWBdata.resp = uwb_resp;
UWBdata.hr = uwb_hr;
UWBdata.state = state;
UWBdata.filtered_sig = DATA_VECTOR_NORMALIZATION(filtered_sig);
UWBdata.baseband_sig = DATA_VECTOR_NORMALIZATION(baseband_sig);
UWBdata.doppler_map = doppler_map;
UWBdata.br_idx = br_idx;
UWBdata.Fs_uwb = Fs_uwb;

cd(['E:\Research Data\UWB_Bed_Biopac_dataset\', sprintf('%s',sub_name{sub_n})]);
save UWBdata.mat UWBdata


%% Biopac & UWB sync

clearvars -except sub_n sub_name UWBdata

cd(['E:\Research Data\UWB_Bed_Biopac_dataset\', sprintf('%s',sub_name{sub_n})]);
load UWBdata.mat
load BrLab_Biopac_SyncData.mat

Fs_biopac = Final_SyncData.Fs_biopac;
Biopac_resp = Final_SyncData.ref_resp;

Fs_uwb = UWBdata.Fs_uwb;
UWB_filtdata = UWBdata.filtered_sig;
UWB_brIdx = UWBdata.br_idx;

selIdx = mode(UWB_brIdx(find(UWB_brIdx<400)));
UWB_resp = UWB_filtdata(selIdx,:);

% Filtering & Normalization
[b1, a1] = butter(5, 0.5/(Fs_uwb/2), 'low');
[b2, a2] = butter(5, 0.1/(Fs_uwb/2), 'high');

f_UWB_resp = filtfilt(b1, a1, UWB_resp);
UWB_resp = filtfilt(b2, a2, f_UWB_resp);

biopac_time = [1:length(Biopac_resp)];
biopac_time = biopac_time/Fs_biopac;

uwb_time = [1:length(UWB_resp)];
uwb_time = uwb_time/Fs_uwb;


fig = figure;
fig.WindowState = 'maximized';
subplot(311); plot(uwb_time, UWB_resp, 'b'); axis tight; title('UWB respiration'); xlim([1 1000]);
subplot(312); plot(biopac_time, Biopac_resp, 'r'); axis tight; title('Biopac respiration'); xlim([1 1000]);
subplot(313); plot(uwb_time, UWB_resp, 'b'); hold on; xlim([1 1000]);
plot(biopac_time, Biopac_resp, 'r'); axis tight; xlim([1 1000]); legend('UWB', 'Biopac');


% UWB 가 뒤에 있다고 가정
XLIM = [];  YLIM = [];

time_shift = 0;        % default : 0 .  time_shift = -3;
time_shift_sum = 0;
re_biopac_time = biopac_time + time_shift;


%% Figure 1 그림에서 3군데 확대 후 아래 섹션 3번 실행(한번씩 하고 실행할 것)

h = subplot(313);

XLIM = [XLIM; h.XLim];
YLIM = [YLIM; h.YLim];

time_shift = 0;        % default : 0 .  time_shift = -100;
time_shift_sum = 0;
re_biopac_time = biopac_time + time_shift;

%% Time shift 값을 설정 후, Biopac과 UWB 싱크가 맞춰지는 구간까지 아래 섹션을 반복 실행하기

% time_shift = -1/2;         time_shift = -1;          time_shift = -5;         time_shift = -10;
% time_shift = +1/2;         time_shift = +1;          time_shift = +5;         time_shift = +10;

time_shift = +1/4;
time_shift_sum = time_shift_sum + time_shift;
re_biopac_time = re_biopac_time + time_shift;

figure(2); clf;
for k = 1 : 3

    subplot(3,1,k)
    plot(uwb_time, UWB_resp, 'r');
    hold on; plot(re_biopac_time, Biopac_resp, 'b');
    xlim([XLIM(k,1) XLIM(k,2)]); legend('UWB', 'Biopac')
    %     ylim([YLIM(k,1) YLIM(k,2)]);

end

%% 최종확인

figure(3); clf;

subplot(211); plot(uwb_time, UWB_resp, 'r'); axis tight;
hold on; plot(biopac_time, Biopac_resp, 'b');  title('Before shift'); legend('UWB', 'Biopac')

subplot(212); plot(uwb_time, UWB_resp, 'r'); axis tight;
hold on; plot(re_biopac_time, Biopac_resp, 'b'); title('After shift'); legend('UWB', 'Biopac')


%%

close all;

UWB_Biopac_SyncData.Fs_uwb = Fs_uwb;
UWB_Biopac_SyncData.Fs_biopac = Fs_biopac;
UWB_Biopac_SyncData.biopac_resp = Final_SyncData.ref_resp;
UWB_Biopac_SyncData.biopac_ecg = Final_SyncData.ref_ecg;

% 시작점 맞추기
if re_biopac_time(1) < uwb_time(1)
    st_time = uwb_time(1) - re_biopac_time(1);
    UWB_Biopac_SyncData.biopac_resp = UWB_Biopac_SyncData.biopac_resp(st_time*Fs_biopac+1 : end);
    UWB_Biopac_SyncData.biopac_ecg = UWB_Biopac_SyncData.biopac_ecg(st_time*Fs_biopac+1 : end);
else
    st_time = re_biopac_time(1) - uwb_time(1);
    UWB_Biopac_SyncData.uwb_rawSig = UWBdata.rawdata(:, st_time*Fs_uwb+1 : end);
    UWB_Biopac_SyncData.uwb_filtSig = UWBdata.filtered_sig(:, st_time*Fs_uwb+1 : end);
    UWB_Biopac_SyncData.uwb_basebSig = UWBdata.baseband_sig(:, st_time*Fs_uwb+1 : end);
    UWB_Biopac_SyncData.uwb_dopplerMap = UWBdata.doppler_map(:, st_time*Fs_uwb+1 : end);
    UWB_Biopac_SyncData.uwb_brIdx = UWBdata.br_idx(:, st_time*Fs_uwb+1 : end);   
end

% 끝점 맞추기
if length(UWB_Biopac_SyncData.biopac_resp)/UWB_Biopac_SyncData.Fs_biopac < length(UWB_Biopac_SyncData.uwb_filtSig(1,:))/UWB_Biopac_SyncData.Fs_uwb
    end_time = length(UWB_Biopac_SyncData.biopac_resp)/UWB_Biopac_SyncData.Fs_biopac;

    UWB_Biopac_SyncData.uwb_rawSig = UWB_Biopac_SyncData.uwb_rawSig(:, 1 : fix(end_time*UWB_Biopac_SyncData.Fs_uwb));
    UWB_Biopac_SyncData.uwb_filtSig = UWB_Biopac_SyncData.uwb_filtSig(:, 1 : fix(end_time*UWB_Biopac_SyncData.Fs_uwb));
    UWB_Biopac_SyncData.uwb_basebSig = UWB_Biopac_SyncData.uwb_basebSig(:, 1 : fix(end_time*UWB_Biopac_SyncData.Fs_uwb));
    UWB_Biopac_SyncData.uwb_dopplerMap = UWB_Biopac_SyncData.uwb_dopplerMap(:, 1 : fix(end_time*UWB_Biopac_SyncData.Fs_uwb));
    UWB_Biopac_SyncData.uwb_brIdx = UWB_Biopac_SyncData.uwb_brIdx(:, 1 : fix(end_time*UWB_Biopac_SyncData.Fs_uwb));   
else
    end_time = length(UWB_Biopac_SyncData.uwb_filtSig(1,:))/UWB_Biopac_SyncData.Fs_uwb
    UWB_Biopac_SyncData.biopac_resp = UWB_Biopac_SyncData.biopac_resp(1 : fix(end_time*UWB_Biopac_SyncData.Fs_biopac));
    UWB_Biopac_SyncData.biopac_ecg = UWB_Biopac_SyncData.biopac_ecg(1 : fix(end_time*UWB_Biopac_SyncData.Fs_biopac));
end

save('UWB_Biopac_SyncData.mat', 'UWB_Biopac_SyncData');


%%%%%%%%%%%%%%%%%%%%%% Sync figure check %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


biopac_time = [1:length(UWB_Biopac_SyncData.biopac_resp)];
biopac_time = biopac_time/UWB_Biopac_SyncData.Fs_biopac;

uwb_time = [1:length(UWB_Biopac_SyncData.uwb_filtSig(selIdx,:))];
uwb_time = uwb_time/UWB_Biopac_SyncData.Fs_uwb;

figure; clf;
subplot(311); plot(uwb_time, UWB_Biopac_SyncData.uwb_filtSig(selIdx,:), 'r'); axis tight; xlim([1 100]);
hold on; plot(biopac_time, UWB_Biopac_SyncData.biopac_resp, 'b'); axis tight; xlim([1 100]); legend('UWB', 'Biopac');
subplot(312); plot(uwb_time,UWB_Biopac_SyncData.uwb_filtSig(selIdx,:), 'r'); axis tight; xlim([350 450]);
hold on; plot(biopac_time, UWB_Biopac_SyncData.biopac_resp, 'b'); axis tight; xlim([350 450]); legend('UWB', 'Biopac');
subplot(313); plot(uwb_time,UWB_Biopac_SyncData.uwb_filtSig(selIdx,:), 'r'); axis tight; xlim([800 900]);
hold on; plot(biopac_time, UWB_Biopac_SyncData.biopac_resp, 'b'); axis tight; xlim([800 900]); legend('UWB', 'Biopac');






