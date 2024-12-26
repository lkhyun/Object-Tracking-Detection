function [curState, noHumanCNT, noMoveCNT, b_MovementBEG, peakMoveEnergy] = ActivityClassification_v5(prevState, TotalEnergy, firstMoveEnergy, lastMoveEnergy, peakMoveEnergy, Threshold_NoHuman, Threshold_Movement, Thresh_ER, framePerMeasure, noHumanCNT, noMoveCNT, b_MovementBEG)
STATE_NOHUMAN = 0;  STATE_MOVEMENT = 1; STATE_STABLE = 2;   
if TotalEnergy < Threshold_NoHuman,
    noHumanCNT = noHumanCNT + 1;
    if noHumanCNT >= framePerMeasure,                   % If signal strength is less than Threshold_NoHuman for more than 6 seconds
        curState = STATE_NOHUMAN;                       % No human status determined
    else            
        curState = prevState;                           % Maintain previous status
    end
else % If there exist a human,...
    noHumanCNT = 0;                                     % Reset No human count
    % Threshold calculation: 6*x/((6-t)+t*x) 
    if lastMoveEnergy > TotalEnergy * Thresh_ER,       % abrupt Energy Increase detected!! x5 times (x = 5)
        if ~b_MovementBEG,
            peakMoveEnergy = lastMoveEnergy;        
        else
            if lastMoveEnergy > peakMoveEnergy,
                peakMoveEnergy = lastMoveEnergy;
            end
        end            
        b_MovementBEG = 1;                              % Start of Movement period
        noMoveCNT = 0;
        curState = STATE_MOVEMENT;                
    elseif firstMoveEnergy > TotalEnergy * Thresh_ER,  % abrupt Energy Decrease detected x 1/5 times (x = 5)            
        b_MovementBEG = 0;                 
        noMoveCNT = 0;                
        curState = STATE_MOVEMENT;            
    elseif TotalEnergy > Threshold_Movement,
        noMoveCNT = 0;
        curState = STATE_MOVEMENT;
%     elseif lastMoveEnergy > peakMoveEnergy * 0.1,
%         b_MovementBEG = 0;                 
%         noMoveCNT = 0;                
%         curState = STATE_MOVEMENT;
    else
        noMoveCNT = noMoveCNT + 1;            
        if b_MovementBEG,                
            curState = STATE_MOVEMENT;
            if noMoveCNT >= framePerMeasure,        % 6 seconds                                        
                b_MovementBEG = 0;                    
                noMoveCNT = 0;                
            end                
        else
            if noMoveCNT >= framePerMeasure,        % 6 seconds
                curState = STATE_STABLE;                   
                noMoveCNT = framePerMeasure;                    
            else
                curState = prevState;
            end
        end
    end
end