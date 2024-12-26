% Vector Normalization
function [NormailzedData]  =  DATA_VECTOR_NORMALIZATION(DATA)
%#codegen
[ROW_SIZE, COLUMN_SIZE]  =  size(DATA);                                          % Data size
NormailzedData           =  zeros(ROW_SIZE, COLUMN_SIZE);                        % Data size만큼 배열 초기화
Dir                      =  0;                                                   % Direction

if COLUMN_SIZE > ROW_SIZE                                                        % Column의 size가 크면 
    Channel_Size   =  ROW_SIZE;                                                  % Channel size에 Row값을 받는다
    Dir            =  1;                                                         % Direction: 1
else                                                                             % Row값을 size가 크면
    Channel_Size   =  COLUMN_SIZE;                                               %  Channel size에 Column값을 받는다
    Dir            =  -1;                                                        % Direction: -1
end

for k=1:1:Channel_Size                                                           % Channle size를 돌며 데이터 Normalization 1
    if Dir == 1                                                                  % Channle size를 돌며 데이터 Normalization 2
        NormailzedData(k,:)  = (DATA(k,:) - mean(DATA(k,:)))/ std(DATA(k,:));    % Channle size를 돌며 데이터 Normalization 3
    elseif Dir == -1                                                             % Channle size를 돌며 데이터 Normalization 4
        NormailzedData(:,k)  = (DATA(:,k) - mean(DATA(:,k)))/ std(DATA(:,k));    % Channle size를 돌며 데이터 Normalization 5
    end                                                                          % Channle size를 돌며 데이터 Normalization 6
end                                                                              % Channle size를 돌며 데이터 Normalization 7
     
    