function [Pxx,F] = zoomfft(x,fs,fe,Fs,fftsize)

len_f = 128;
% Determine Decimation Number
delta_f = fe-fs;
fc = (fs+fe)/2;
t = 0:(length(x)-1);
t = t/Fs;
if size(t,1) ~= size(x,1), t = t'; end

% Deci_N = floor(Fs/2 /delta_f);
Deci_N = floor(Fs/delta_f);
Deci_N = round(Deci_N/10)*10;
% Deci_N = 20;
% LPF design
b=fir1(len_f,1/Deci_N);
shiftFunc = exp(-1i*2*pi*fc*t);
shiftSig = x.*shiftFunc;

% FilterSignal=filter(b,1,shiftSig);
FilterSignal = conv(shiftSig,b);
FilterSignal = FilterSignal(len_f/2+1:end-len_f/2);
M=length(FilterSignal);
D = Deci_N;
DownLength=1:M/D;                                               %downsample signal length.
DownsampleSignal(DownLength)=FilterSignal((DownLength-1)*D+1);  %downsample the signal

% fftwin2 = hamming(length(DownsampleSignal));
% DownsampleSignal = DownsampleSignal.*fftwin2';
y = abs(fft(DownsampleSignal,fftsize)).^2;
FsD = Fs/Deci_N;
FRes = FsD/fftsize;
fcD = ceil(delta_f/FRes/2);
Pxx = [y(end-fcD+1:end) y(1:fcD)];
F = [(fc-FRes*fcD:FRes:fc-FRes) (fc:FRes:fc+FRes*(fcD-1))];