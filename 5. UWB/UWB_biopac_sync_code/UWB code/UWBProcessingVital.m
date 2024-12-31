classdef UWBProcessingVital
    properties        
        targetPRF
        numCounters
        fftsize = 512;        
        offsetIdx = 10;         % Initial (25 * 0.4cm = 10 cm ) is ignored
        QueInSec = 6;           % time window for vital analysis  
        QueInSecDoppler = 6;    % time window for u-Doppler
        DisplayInSec = 6;
        timeInSec_MoveWin = 3;  % time window for activity energy calculation
        ffttimeInSec =12        % time window for FFT of vital analysis
        
        framePerMeasure
        movementWinLen
        fftdata_len
        InitializingFrames
        SkipSamples
        timeStamp
        
       %% Doppler Parameters
        fftsize_doppler = 256;
        decimation = 8
        framePerDoppler        
        spectral_mul                
        spectral_offset
        spectral_new
        fftWin
        
       %% Actitivity Classification Parameters
        Threshold_NoHuman = 5e-8;           % Threshold for human existence detection 
        Threshold_Movement = 300;           % Threshold for large movement detection
        Threshold_EnergyRatio = 5;          % Threshold. of Energy Ratio for movement detection
        Thresh_ER
        curState
        prevState
        b_MovementBEG = 0;
        b_StableBEG = 0;
        noMoveCNT = 0;
        noHumanCNT = 0;
        buffSTidx = 1;
        peakMoveEnergy;
        energyProfile = [0 0 0];
        
       %% Breathing Rate Parameters
        lowerlimit_BR = 6/60;
        higherlimit_BR = 60/60;
        
       %% Heart Rate Parameters
        lowerlimit_HR = 42/60;
        higherlimit_HR = 150/60;
       
       %% Buffer 
        m_OrigData
        m_ProcessingData
        m_BasebandData
        a_EnergyFrame
        a_ColumnHistory
        
        m_DopplerFFT        
        m_DopplerMap
        a_ClutterBaseband
        a_ClutterBaseband_pre
        a_ClutterProcessing
        a_ClutterProcessing_pre        
        a_ClutterORG
        a_ClutterORG_pre
        a_ClutterDynamic
        baseBand
        a_RmeanEnergy
        
        RespSig        
        RespSig_BUFF
        HRSig
        
        
       %% Filter 
        Alpha = 0.96;       % Clutter update coefficient
        M = 32;             % LMS filter tab
        mu = 0.01;          % LMS filter coefficient
        w
        
       %% Basic Filter Generation                
        ydemod
        b_baseband
        stidxBB
        frameCNT          
        Resolution
        
        yi
        offset
        
       %% STATUS parameters
        STATE_NOHUMAN = 0;  
        STATE_MOVEMENT = 1; 
        STATE_STABLE = 2;   
        STATE_EMERGENCY = 3;
        
       %% Result
        brIdx = 1;
        estimated_HR = 0;
        estimated_BR = 0;
        
    end
    
    methods 
        function obj = UWBProcessingVital(fps, numCounters , uwbVer)
             
           %% Basic Variables
            obj.targetPRF = fps;                     % Target Frame rate: 20Hz (default)                                                                       
            obj.numCounters = numCounters;
            
            obj.curState = obj.STATE_MOVEMENT;
            obj.prevState = obj.STATE_MOVEMENT;


           %% Dependant Variables            
            obj.framePerMeasure = obj.QueInSec*obj.targetPRF;      
            obj.movementWinLen = obj.timeInSec_MoveWin*obj.targetPRF;
            obj.fftdata_len = obj.ffttimeInSec*obj.targetPRF;
            obj.InitializingFrames = 3*obj.targetPRF;  
            obj.SkipSamples = 0.5*obj.targetPRF;
            obj.framePerDoppler = obj.QueInSecDoppler*obj.targetPRF;
            obj.Thresh_ER = obj.QueInSec*obj.Threshold_EnergyRatio/((obj.QueInSec-obj.timeInSec_MoveWin)+obj.timeInSec_MoveWin*obj.Threshold_EnergyRatio);

           %% Adaptive Filter coefficient for LMS filter            
            obj.w = zeros(obj.M,1);
            
           %% Variables for DFT          
            k = 0:obj.fftsize_doppler-1;
            obj.spectral_mul = transpose(exp(1i*2*pi*k/obj.fftsize_doppler));
            % spectral_mul(1) = 0;
            m = 0:obj.framePerMeasure-1;
            spectral_mul2 = exp(-1i*2*pi*k'*m/obj.fftsize_doppler);
            obj.spectral_offset = sum(spectral_mul2,2);
            obj.spectral_new = spectral_mul2(:,end);
            
            obj.fftWin = hamming(obj.fftdata_len);
            
           %% Filter for Baseband
            if strcmp(uwbVer,'x2')
                fs_uwb = 3.7500e+10;
                fc = 7.7e9;
                bw = 2.5e9;
            elseif strcmp(uwbVer,'x4')
                fs_uwb = 23.328e9;
                fc = 7.29e9;      
                bw = 1.4e9;
            else
                fs_uwb = 23.328e9;
                fc = 7.29e9;
                bw = 1.4e9;
            end                
            obj.Resolution = 3e8/fs_uwb/2;
            obj.ydemod = exp(-1i*2*pi*(0:obj.numCounters-1)*fc/fs_uwb);
%             obj.b_baseband=[ 0.0022 0.0062 0.0129 0.0226 0.0352 0.0500 0.0656 0.0804 0.0927 ...
%             0.1008 0.1036 0.1008 0.0927 0.0804 0.0656 0.0500 0.0352 0.0226 ...
%             0.0129 0.0062 0.0022]*sqrt(2);
            obj.b_baseband = fir1(21,3e9/(fs_uwb/2),'low')*sqrt(2);  % 20200723 update

            obj.stidxBB = round(length(obj.b_baseband)/2);           
            
            tc = gauspuls('cutoff',fc,bw/fc,-10,-60);
            resol = obj.Resolution;
            t_sr = resol*2/3e8;
            t = -tc:t_sr:tc;
            vt = gauspuls(t,fc,bw/fc,-10);
            offset = floor(length(vt)/2);
            obj.yi = vt;
            obj.offset = offset;
            
            
           %% Buffer 
            obj.m_OrigData = zeros(obj.framePerMeasure,obj.numCounters);  
            obj.m_ProcessingData = zeros(obj.framePerMeasure,obj.numCounters);
            obj.m_BasebandData = zeros(obj.framePerMeasure,obj.numCounters/obj.decimation);           % Collaboration Matrix
            obj.a_EnergyFrame = zeros(obj.framePerMeasure,1);
            obj.a_ColumnHistory = zeros(obj.framePerMeasure,5);
            obj.a_ColumnHistory(:,1:4) = 1;
                                            
            obj.m_DopplerMap = zeros(obj.fftsize_doppler, obj.targetPRF*obj.DisplayInSec);
            obj.m_DopplerFFT = zeros(obj.fftsize_doppler,obj.numCounters/obj.decimation);
            obj.a_ClutterBaseband = zeros(1,obj.numCounters/obj.decimation);
            obj.a_ClutterBaseband_pre = zeros(1,obj.numCounters/obj.decimation);
            obj.a_ClutterProcessing = zeros(1,obj.numCounters);
            obj.a_ClutterProcessing_pre = zeros(1,obj.numCounters);            
            obj.a_ClutterORG = zeros(1,obj.numCounters);
            obj.a_ClutterORG_pre = zeros(1,obj.numCounters);            
            obj.a_ClutterDynamic = zeros(1,obj.numCounters);
            obj.a_RmeanEnergy = zeros(1,obj.numCounters);
            
            obj.RespSig = zeros(obj.fftdata_len,1);
            obj.RespSig_BUFF = zeros(obj.fftdata_len,1);
            obj.HRSig = zeros(obj.fftdata_len,1);
            
            obj.timeStamp = zeros(obj.fftdata_len,1);
            
            obj.brIdx = obj.offsetIdx;
            
            obj.frameCNT = 1;
        end
        function obj = process(obj, RawData, interval)
           
%             sprintf('CNT: %d', obj.frameCNT) %% 20200720 
            obj.timeStamp(1:end-1) = obj.timeStamp(2:end);
            obj.timeStamp(end) = interval;
           %% 1. Preprocessing, Clutter removal & Collaboration Matrix construction    
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Preprocessing Baseband Extraction
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Baseband = (RawData-mean(RawData)).*obj.ydemod;    
            TTT = conv(Baseband,obj.b_baseband);
            fBaseband = TTT(obj.stidxBB:obj.stidxBB+obj.numCounters-1);      
            obj.baseBand = abs(fBaseband);      
            
%             a_xcorr = xcorr(RawData-mean(RawData),obj.yi);
%             obj.baseBand = a_xcorr(obj.numCounters-obj.offset:end-obj.offset);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Dynamic Clutter Estimation
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if obj.frameCNT == 1
                obj.a_ClutterDynamic = obj.baseBand;
            else
                obj.a_ClutterDynamic = obj.Alpha*obj.a_ClutterDynamic + (1-obj.Alpha)*obj.baseBand;
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Update Collaboration Matrix for Baseband
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                                
            a_Baseband_out = obj.m_BasebandData(1,:);
            obj.m_BasebandData(1:end-1,:) = obj.m_BasebandData(2:end,:);    
            a_Baseband_in = fBaseband(1:obj.decimation:end);            
            obj.m_BasebandData(end,:) = a_Baseband_in;               
            
            % Clutter for base band
            if obj.frameCNT <= obj.framePerMeasure
                numFrame = obj.frameCNT;
                obj.a_ClutterBaseband = (obj.a_ClutterBaseband*(numFrame-1) + a_Baseband_in)/numFrame;        
            else        
                obj.a_ClutterBaseband = obj.a_ClutterBaseband + (a_Baseband_in-a_Baseband_out)/obj.framePerMeasure;        
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Update Collaboration Matrix for Processing Data
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            a_Processing_out = obj.m_ProcessingData(1,:);
            obj.m_ProcessingData(1:end-1,:) = obj.m_ProcessingData(2:end,:);
            a_Processing_in = obj.baseBand - obj.a_ClutterDynamic;
            obj.m_ProcessingData(end,:) = a_Processing_in;
            
           %% 2. Energy Characteristics Analysis    
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Energy Profile Calculation
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            % Spatial Energy Profile
            if obj.frameCNT <= obj.framePerMeasure
                obj.a_RmeanEnergy = (obj.a_RmeanEnergy*(obj.frameCNT-1) + a_Processing_in.^2)/obj.frameCNT;
            else
                obj.a_RmeanEnergy = obj.a_RmeanEnergy + (a_Processing_in.^2 - a_Processing_out.^2)/obj.framePerMeasure;
            end
            
            % Temporal Energy Profile
            obj.a_EnergyFrame(1:end-1) = obj.a_EnergyFrame(2:end);
            obj.a_EnergyFrame(end) = sum(a_Processing_in(obj.offsetIdx:end).^2);
%             obj.a_EnergyFrame(end) = sum(a_Processing_in.^2);
            TotalEnergy = mean(obj.a_EnergyFrame);
            lastMoveEnergy = mean(obj.a_EnergyFrame(end-obj.movementWinLen+1:end));
            firstMoveEnergy = mean(obj.a_EnergyFrame(1:obj.movementWinLen));            
            
            obj.energyProfile = [TotalEnergy lastMoveEnergy firstMoveEnergy];
            
          %% 3. Make Doppler map
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Micro-Doppler Map 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         
            if obj.frameCNT > obj.framePerMeasure
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Real time code start here!!  (using Zero-padding Sliding DFT)       
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Sliding DFT
                col_idx = obj.frameCNT-obj.framePerMeasure;
                if col_idx > obj.numCounters/obj.decimation % framePerMeasure,
                    % Update DFT
                    for i=1:size(obj.m_DopplerFFT,2)
                        X_prev = obj.m_DopplerFFT(:,i);
                        X_cur = (X_prev-(a_Baseband_out(i)-obj.a_ClutterBaseband_pre(i))).*obj.spectral_mul + (a_Baseband_in(i)-obj.a_ClutterBaseband_pre(i))*obj.spectral_new;
                        X_cur = X_cur + (obj.a_ClutterBaseband_pre(i)-obj.a_ClutterBaseband(i))*obj.spectral_offset;
                        X_cur(1) = 0;
                        obj.m_DopplerFFT(:,i) = X_cur;                
                    end
                else
                    % Initial FFT
                    temp = obj.m_BasebandData(:,col_idx) - obj.a_ClutterBaseband(col_idx);
                    fft_temp = fft(temp,obj.fftsize_doppler);
                    obj.m_DopplerFFT(:,col_idx) = fft_temp;                    
                    for i=1:col_idx-1
                        X_prev = obj.m_DopplerFFT(:,i);
                        X_cur = (X_prev-(a_Baseband_out(i)-obj.a_ClutterBaseband_pre(i))).*obj.spectral_mul + (a_Baseband_in(i)-obj.a_ClutterBaseband_pre(i))*obj.spectral_new;
                        X_cur = X_cur + (obj.a_ClutterBaseband_pre(i)-obj.a_ClutterBaseband(i))*obj.spectral_offset;
                        X_cur(1) = 0;                
                        obj.m_DopplerFFT(:,i) = X_cur;                
                    end
                end
                obj.a_ClutterBaseband_pre = obj.a_ClutterBaseband;
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Build u-Doppler Signature
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                     
                m_DopplerFFTABS = abs(obj.m_DopplerFFT);
%                 m_FFTAll = m_DopplerFFTABS;
                m_FFTAll = [m_DopplerFFTABS(obj.fftsize_doppler/2:end,:); m_DopplerFFTABS(1:obj.fftsize_doppler/2-1,:)];        
                a_Doppler = sum(m_FFTAll,2);
                obj.m_DopplerMap(:,1:end-1) = obj.m_DopplerMap(:,2:end);
                obj.m_DopplerMap(:,end) = a_Doppler;    
                
                if col_idx > obj.numCounters/obj.decimation
                    idx_1Hz = floor(obj.fftsize_doppler/obj.targetPRF);
                    pw_under1Hz = sum(m_DopplerFFTABS(2:idx_1Hz,:));
                    pw_ratio = pw_under1Hz./sum(m_DopplerFFTABS(2:obj.fftsize_doppler/2,:));
                    pw_R = interp(pw_ratio,obj.decimation);
                else
                    pw_R = ones(size(obj.a_RmeanEnergy));
                end
                
              %% 4. Activity Classification
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Classification: Movement or Stable or No human
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                [obj.curState, obj.noHumanCNT, obj.noMoveCNT, obj.b_MovementBEG, obj.peakMoveEnergy] = ActivityClassification_v5(obj.prevState, TotalEnergy, ...
                    firstMoveEnergy, lastMoveEnergy, obj.peakMoveEnergy, obj.Threshold_NoHuman, obj.Threshold_Movement, obj.Thresh_ER, obj.framePerMeasure, obj.noHumanCNT, obj.noMoveCNT, obj.b_MovementBEG);
                
%                 obj.curState = 2; % Movement removing check, 20200723
                
                a_RmeanEnergyWeighted = obj.a_RmeanEnergy.*pw_R.^2;
                [~, maxIdx] = max(a_RmeanEnergyWeighted(obj.offsetIdx:end));
                maxIdx = maxIdx + obj.offsetIdx - 1;
                maxAmp = obj.a_RmeanEnergy(maxIdx);
                if obj.frameCNT> 480
                    bp = 1;
                end
                
                if obj.curState == obj.STATE_STABLE                
              %% 5. Do Vital Sign Monitoring
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Respiration Column Detection
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    ColumnShift_mode = 0;
                    if obj.prevState == obj.STATE_MOVEMENT
                        % State Changed from Movement to Stable
                        ColumnShift_mode = 2;
                        brIdx = maxIdx;
                        % Reset Column History 
                        obj.a_ColumnHistory(:,1:4) = 1;                         
                        % Reset Vital Signal Buffer and Append current respiration signal to Buffer
                        obj.RespSig_BUFF(:) = 0;
                        obj.RespSig_BUFF(end-obj.framePerMeasure+1:end) = obj.m_ProcessingData(:,maxIdx);
                    else
                        % Update Column History
                        obj.a_ColumnHistory(1:end-1,:) = obj.a_ColumnHistory(2:end,:);   
                        [brIdx, ColumnShift_mode] =  FindBreathingColumn_v5(obj.a_RmeanEnergy, pw_R, maxIdx, maxAmp, obj.a_ColumnHistory, obj.framePerMeasure);
                        % Update Vital Signal Buffer and Append current respiration signal to Buffer
                        obj.RespSig_BUFF(1:end-1) = obj.RespSig_BUFF(2:end);
                        obj.RespSig_BUFF(end-obj.framePerMeasure+1:end) = obj.m_ProcessingData(:,maxIdx); %????????                        
                    end
                    % Update Column History
                    obj.brIdx = brIdx;
                    obj.a_ColumnHistory(end,1) = maxAmp;                     
                    obj.a_ColumnHistory(end,2) = maxIdx;                     
                    obj.a_ColumnHistory(end,3) = obj.a_RmeanEnergy(brIdx);                   
                    obj.a_ColumnHistory(end,4) = brIdx; 
                    
                    
                 %% Respiration Signal Reconstruction for FFT
                    
                    a_Resp = obj.m_ProcessingData(:,brIdx);
                    switch ColumnShift_mode
                        case 2 % Big Change (movement to stable)
                            obj.buffSTidx = length(obj.RespSig)-obj.framePerMeasure+1;
                            % Reset Signal Buffer and append current signal
                            obj.RespSig(:) = 0;
                            obj.RespSig(end-obj.framePerMeasure+1:end) = a_Resp;
                            
                        case 1 % small change in column during stable period
                            obj.buffSTidx = 1;
                            obj.RespSig = obj.RespSig_BUFF;
                        otherwise
                            if obj.buffSTidx > 1
                                obj.buffSTidx = obj.buffSTidx - 1;
                            end
                            obj.RespSig(1:end-1) = obj.RespSig(2:end);
                            obj.RespSig(end) = a_Resp(end);
                    end
                    
                    
                 %% Preprocessing for HR estimation: LMS filtering
                    
                    obj.HRSig(1:end-1) = obj.HRSig(2:end);
                    if ColumnShift_mode > 0
                        % Reset HR signal
                        obj.HRSig(:) = 0;
                        for l_i = obj.M:obj.fftdata_len
                            stidx = l_i-obj.M+1;
                            edidx = l_i;
                            u = obj.RespSig(stidx:edidx);                        
                            HRCur = obj.RespSig(edidx);
                            y = obj.w'*u;
                            e = HRCur - y;
                            obj.w = obj.w + obj.mu*e*u/(u'*u+eps);
                            obj.HRSig(l_i) = e;                        
                        end 
                    else
                        u = obj.RespSig(end-obj.M+1:end);                    
                        HRCur = obj.RespSig(end);
                        y = obj.w'*u;        
                        e = HRCur - y;
                        obj.w = obj.w + obj.mu*e*u/(u'*u+eps);
                        obj.HRSig(end) = e;                        
                    end
                    
                    
                 %% Frequency Analysis for BR & HR estimation                    
                    % make Analysis Start Flag enable when state has changed
                    if obj.prevState == obj.STATE_MOVEMENT, obj.b_StableBEG = 1; end
                    
                    % Analyze BR & HR every 0.5 seconds if buffer is full(buffSTidx == 1)
                    if (mod(obj.frameCNT,obj.SkipSamples) == 0) && (obj.buffSTidx == 1)                        
                        % Frame Rate Calculation from the record                        
                        Fs = 1000/mean(obj.timeStamp);
                        
                        if obj.frameCNT > 3000,
                            my_bp = 1;
                        end
                                                
                        % DC removal & Windowing
                        zHRSig = obj.HRSig - mean(obj.HRSig);
                        zHRSig = zHRSig.*obj.fftWin;
                        zRespSig = obj.RespSig - mean(obj.RespSig);
                        zRespSig = zRespSig.*obj.fftWin;
                                                
                        % Zoom FFT
                        [z1, freqLocal1] = zoomfft(zHRSig,obj.lowerlimit_HR, obj.higherlimit_HR, Fs, obj.fftsize);
                        [z2, freqLocal2] = zoomfft(zRespSig,obj.lowerlimit_BR, obj.higherlimit_BR, Fs, obj.fftsize);
                        
                        % HR Calculation
                        [~,tidx] = max(z1);
                        obj.estimated_HR = freqLocal1(tidx)*60;   
                        
                        % Breathing Rate Calculation
                        [~,tidx] = max(z2);
                        obj.estimated_BR = freqLocal2(tidx)*60;                        
                        
                    end
                    
                else
              %% Exception: Not Stable Period    
                end
                obj.prevState = obj.curState;
                
            end
            
            obj.frameCNT = obj.frameCNT + 1;               
                        
        end
        
        
    end
end