function [brIdx, b_CShift] = FindBreathingColumn_v5(a_RmeanEnergy, pw_R, brIdx, brAmp, c_R, framePerMeasure)

preIdx = c_R(end-1,4);
b_CShift = 0;

changed_idx = abs(brIdx-preIdx);        
if changed_idx > 0,                         % Column Change Detected!!
    % Compare peak amplitude between previous idx and current idx   
    if brIdx>100
        bp=1;
    end
    peak_ratio = a_RmeanEnergy(preIdx)/brAmp;
%     if (peak_ratio>0.5) || (pw_R(brIdx) < 0.3) % Add condition for pw_R from v24_5
    if (peak_ratio>0.5) || (pw_R(preIdx) > pw_R(brIdx)) % Add condition for pw_R from v24_5
        b_CShift = 0;
    else
        b_CShift = 2;
    end
    cidx = find(abs(c_R(:,2)-brIdx) < 2);
    if (length(cidx) == framePerMeasure),    % If column change continues more than 6 seconds
        if pw_R(brIdx) > pw_R(preIdx),
            b_CShift = 1;
        end
    end
    
    if ~b_CShift,     
        brIdx = preIdx;    
    end
    
    
    
    
end