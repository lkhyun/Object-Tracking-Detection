
%% BrLab sensor data

clc; close all; clear all

sub = '3';
dir_name = ['C:\Users\fgh75\OneDrive - 광운대학교\바탕 화면\학부연구생\rader\ex_dataset\[2401~2402] UWB_Biopac_Bed dataset\', sprintf('%s',sub)];

Final_SyncData.Fs_pvdf = 250;
Final_SyncData.Fs_fsr = 10;
for device_num = 1 : 4
    
    cd(sprintf('%s/PVDF_FSR',dir_name))
    current_dir = dir(sprintf('%s/PVDF_FSR',dir_name));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PVDF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    idx = find(~cellfun(@isempty, strfind({current_dir.name}, 'pvdf_raw')));
    fname = current_dir(idx).name
    fid = fopen(fname, "r");
    pvdf = fread(fid, Inf, "int32");
    fclose(fid);
    Final_SyncData.pvdf_all{1,device_num} = pvdf;

    brlab_time = str2num(fname(7:8))*3600 + str2num(fname(9:10))*60 + str2num(fname(11:12));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FSR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    idx = find(~cellfun(@isempty, strfind({current_dir.name}, 'fsr70_raw')));
    fname = current_dir(idx).name
    fid = fopen(fname, "r");
    data = fread(fid, Inf, "uint16");
    fclose(fid);

    tmp_fsr = reshape(data', [7, length(data)/7]);
    fsr = zeros(7, length(data)/7);
    for sec05 = 1 : length(data)/7/5
        fsr(:, (sec05 - 1)*5  + 1 : (sec05 - 1)*5  + 5) = tmp_fsr(:, (sec05 - 1)*5  + 1 : (sec05 - 1)*5  + 5);
    end
    Final_SyncData.fsr_all{1,device_num} = fsr;

end


%% 

%%%%%%%%%%%%%%%%%%%%%%%%%% Reference biopac data %%%%%%%%%%%%%%%%%%%%%%%%%%
cd(sprintf('%s/Bed_Radar/',dir_name))
current_dir = dir(sprintf('%s/Bed_Radar/',dir_name));
idx = find(~cellfun(@isempty, strfind({current_dir.name}, 'mat')));
fname = current_dir(idx).name
load(fname)

Final_SyncData.Fs_biopac = 250;
Final_SyncData.ref_resp = data(:,1);
Final_SyncData.ref_ecg = data(:,2);

biopac_time = str2num(fname(end-11:end-10))*3600 + str2num(fname(end-8:end-7))*60 + str2num(fname(end-5:end-4));
time_diff = biopac_time - brlab_time; % biopac time - BrLab sensor recording time

if time_diff > 0
    for ch = 1 : 4
        Final_SyncData.pvdf_all{1,ch} = Final_SyncData.pvdf_all{1,ch}(time_diff*Final_SyncData.Fs_pvdf+1 : end);
        Final_SyncData.fsr_all{1,ch} = Final_SyncData.fsr_all{1,ch}(:, time_diff*Final_SyncData.Fs_fsr+1 : end);
    end
else
    Final_SyncData.ref_resp = Final_SyncData.ref_resp(abs(time_diff)*Final_SyncData.Fs_biopac+1:end);
    Final_SyncData.ref_ecg = Final_SyncData.ref_ecg(abs(time_diff)*Final_SyncData.Fs_biopac+1:end);
end


%% 

% ECG filtering
[b1, a1] = butter(5, 10/(Final_SyncData.Fs_pvdf/2), 'low');
[b2, a2] = butter(5, 2/(Final_SyncData.Fs_pvdf/2), 'high');

% RESP filtering
[b3, a3] = butter(5, 0.5/(Final_SyncData.Fs_pvdf/2), 'low');
[b4, a4] = butter(5, 0.1/(Final_SyncData.Fs_pvdf/2), 'high');

for ch = 1 : 4
    tmp_pvdf = Final_SyncData.pvdf_all{1,ch};

    f_tmp_pvdf = filtfilt(b1,a1,tmp_pvdf);
    ff_tmp_pvdf = filtfilt(b2,a2,f_tmp_pvdf);
    Final_SyncData.pvdf_ecg_filt{1,ch} = ff_tmp_pvdf;

    f_tmp_pvdf = filtfilt(b3,a3,tmp_pvdf);
    ff_tmp_pvdf = filtfilt(b4,a4,f_tmp_pvdf);
    Final_SyncData.pvdf_resp_filt{1,ch} = ff_tmp_pvdf;

    fig = figure;
    fig.WindowState = 'maximized';
    subplot(511); plot(tmp_pvdf); axis tight; title(sprintf('Raw PVDF Channel %d', ch)); xlim([2800*Final_SyncData.Fs_pvdf+1 2820*Final_SyncData.Fs_pvdf]);
    subplot(512); plot(Final_SyncData.pvdf_resp_filt{1,ch}); axis tight; title('Filtering 0.1-0.5 Hz'); xlim([2800*Final_SyncData.Fs_pvdf+1 2820*Final_SyncData.Fs_pvdf]);
    subplot(513); plot(Final_SyncData.ref_resp); axis tight; title('Reference RESP'); xlim([2800*Final_SyncData.Fs_biopac+1 2820*Final_SyncData.Fs_biopac]);
    subplot(514); plot(Final_SyncData.pvdf_ecg_filt{1,ch}); axis tight; title('Filtering 2-10 Hz'); xlim([2800*Final_SyncData.Fs_pvdf+1 2820*Final_SyncData.Fs_pvdf]);
    subplot(515); plot(Final_SyncData.ref_ecg); axis tight; title('Reference ECG'); xlim([2800*Final_SyncData.Fs_biopac+1 2820*Final_SyncData.Fs_biopac]);
end

%% 

Final_SyncData.ref_resp = (Final_SyncData.ref_resp - mean(Final_SyncData.ref_resp))./std(Final_SyncData.ref_resp);
for ch = 1 : 4
    Final_SyncData.pvdf_resp_filt{1, ch} = (Final_SyncData.pvdf_resp_filt{1, ch} - mean(Final_SyncData.pvdf_resp_filt{1, ch}))./std(Final_SyncData.pvdf_resp_filt{1, ch});
end
biopac_time = [1:length(Final_SyncData.ref_resp)];
biopac_time = biopac_time/Final_SyncData.Fs_biopac;

pvdf_time = [1:length(Final_SyncData.pvdf_resp_filt{1, 1})];
pvdf_time = pvdf_time/Final_SyncData.Fs_pvdf;


sel_ch = 3; % 위에 섹션에서 PVDF 채널 퀄리티가 좋은 곳 선택

fig = figure;
fig.WindowState = 'maximized';
subplot(311); plot(pvdf_time, Final_SyncData.pvdf_resp_filt{1,sel_ch}, 'b'); axis tight; title('PVDF');
subplot(312); plot(biopac_time, Final_SyncData.ref_resp, 'r'); axis tight; title('Biopac');
subplot(313); plot(pvdf_time, Final_SyncData.pvdf_resp_filt{1,sel_ch}, 'b'); hold on; 
plot(biopac_time, Final_SyncData.ref_resp, 'r'); axis tight; title('biopac ref');  legend('PVDF', 'Biopac');


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

time_shift = -1;
time_shift_sum = time_shift_sum + time_shift;
re_biopac_time = re_biopac_time + time_shift;

figure(2); clf;
for k = 1 : 3

    subplot(3,1,k)
    plot(pvdf_time, Final_SyncData.pvdf_resp_filt{1, sel_ch}, 'r');
    hold on; plot(re_biopac_time, Final_SyncData.ref_resp, 'b');
    xlim([XLIM(k,1) XLIM(k,2)]);  
%     ylim([YLIM(k,1) YLIM(k,2)]);

end

%% 최종확인

figure(3); clf;

subplot(211); plot(biopac_time, Final_SyncData.ref_resp, 'b'); axis tight;
hold on; plot(pvdf_time, Final_SyncData.pvdf_resp_filt{1, sel_ch}, 'r'); title('Before shift'); legend('Biopac', 'PVDF');

subplot(212); plot(re_biopac_time, Final_SyncData.ref_resp, 'b'); axis tight;
hold on; plot(pvdf_time, Final_SyncData.pvdf_resp_filt{1, sel_ch}, 'r'); title('After shift'); legend('Biopac', 'PVDF');


%% 

Fs_pvdf = Final_SyncData.Fs_pvdf;
Fs_fsr = Final_SyncData.Fs_fsr;
Fs_biopac = Final_SyncData.Fs_biopac;

% 시작점 맞추기
if re_biopac_time(1) < pvdf_time(1)
   st_time = pvdf_time(1) - re_biopac_time(1);
   for ch = 1 : 4
       Final_SyncData.pvdf_all{1, ch} = Final_SyncData.pvdf_all{1, ch}(st_time*Fs_pvdf+1 : end);
       Final_SyncData.pvdf_resp_filt{1, ch} = Final_SyncData.pvdf_resp_filt{1, ch}(st_time*Fs_pvdf+1 : end);
       Final_SyncData.pvdf_ecg_filt{1, ch} = Final_SyncData.pvdf_ecg_filt{1, ch}(st_time*Fs_pvdf+1 : end);
       Final_SyncData.fsr_all{1, ch} = Final_SyncData.fsr_all{1, ch}(:, st_time*Fs_fsr+1 : end); 
   end
else
    st_time = re_biopac_time(1) - pvdf_time(1);
    Final_SyncData.ref_resp = Final_SyncData.ref_resp(st_time*Fs_biopac+1 : end);
    Final_SyncData.ref_ecg = Final_SyncData.ref_ecg(st_time*Fs_biopac+1 : end);
end


% 끝점 맞추기
if length(Final_SyncData.ref_ecg) < length(Final_SyncData.pvdf_all{1,1})
    end_idx = length(Final_SyncData.ref_ecg);
    for ch = 1 : 4
       Final_SyncData.pvdf_all{1, ch} = Final_SyncData.pvdf_all{1, ch}(1 : end_idx);
       Final_SyncData.pvdf_resp_filt{1, ch} = Final_SyncData.pvdf_resp_filt{1, ch}(1 : end_idx);
       Final_SyncData.pvdf_ecg_filt{1, ch} = Final_SyncData.pvdf_ecg_filt{1, ch}(1 : end_idx);
       Final_SyncData.fsr_all{1, ch} = Final_SyncData.fsr_all{1, ch}(:, 1 : fix(end_idx/(Fs_pvdf/Fs_fsr))); 
    end
else
    end_idx = length(Final_SyncData.pvdf_all{1,1});
    Final_SyncData.ref_resp = Final_SyncData.ref_resp(1 : end_idx);
    Final_SyncData.ref_ecg = Final_SyncData.ref_ecg(1 : end_idx);
end


cd(dir_name)
save('BrLab_Biopac_SyncData.mat', 'Final_SyncData');


%% Sync figure check

time_x = [1 : length(Final_SyncData.pvdf_resp_filt{1, 1})];
time_x = time_x/Fs_pvdf;

for st_time = 1 : 10 : 900
    end_time = st_time+200

    figure(1); clf;
    subplot(411); plot(time_x, Final_SyncData.pvdf_resp_filt{1, 1}); axis tight; title('PVDF Ch1'); xlim([st_time end_time]);
    hold on; plot(time_x, Final_SyncData.ref_resp, 'r'); axis tight; xlim([st_time end_time]); legend('PVDF', 'Biopac');
    subplot(412); plot(time_x,Final_SyncData.pvdf_resp_filt{1, 2}); axis tight; title('PVDF Ch2'); xlim([st_time end_time]);
    hold on; plot(time_x, Final_SyncData.ref_resp, 'r'); axis tight; xlim([st_time end_time]); 
    subplot(413); plot(time_x,Final_SyncData.pvdf_resp_filt{1, 3}); axis tight; title('PVDF Ch3'); xlim([st_time end_time]);
    hold on; plot(time_x, Final_SyncData.ref_resp, 'r'); axis tight; xlim([st_time end_time]); 
    subplot(414); plot(time_x,Final_SyncData.pvdf_resp_filt{1, 4}); axis tight; title('PVDF Ch4'); xlim([st_time end_time]);
    hold on; plot(time_x, Final_SyncData.ref_resp, 'r'); axis tight; xlim([st_time end_time]); 
    pause(0.5)

end



