% Vector Normalization
function [NormailzedData]  =  COLUMN_DATA_VECTOR_NORMALIZATION(DATA)
%#codegen
[ROW_SIZE, COLUMN_SIZE]  =  size(DATA);                                          % Data size
NormailzedData           =  zeros(ROW_SIZE, COLUMN_SIZE);                        % Data size만큼 배열 초기화
Dir                      =  0;                                                   % Direction

if COLUMN_SIZE > ROW_SIZE                                                        % Column의 size가 크면 
    Time_Size   =  COLUMN_SIZE;                                                  % Channel size에 Row값을 받는다
    Dir            =  1;                                                         % Direction: 1
else                                                                             % Row값을 size가 크면
    Time_Size   =  ROW_SIZE;                                                     %  Channel size에 Column값을 받는다
    Dir            =  -1;                                                        % Direction: -1
end

for k=1:1:Time_Size                                                           
    if Dir == 1    
        NormailzedData(:,k)  = (DATA(:,k) - mean(DATA(:,k)))/ std(DATA(:,k));
        % min_val = min(NormailzedData(:,k));
        % max_val = max(NormailzedData(:,k));
        % % Normalize the data using Min-Max scaling
        % NormailzedData(:,k) = (NormailzedData(:,k) - min_val) / (max_val - min_val);
    elseif Dir == -1                 
        
        NormailzedData(k,:)  = (DATA(k,:) - mean(DATA(k,:)))/ std(DATA(k,:));
        % min_val = min(NormailzedData(k,:));
        % max_val = max(NormailzedData(k,:));
        % % Normalize the data using Min-Max scaling
        % NormailzedData(k,:) = (NormailzedData(k,:) - min_val) / (max_val - min_val);
        
    end                                                                          
end                                                                              
    