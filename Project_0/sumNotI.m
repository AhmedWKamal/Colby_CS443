function [arr] = sumNotI(l)
    arr = zeros(size(l));
    total_sum = sum(l, ["all"]);
    for i = 1:numel(l)
        arr(i) = total_sum - l(i);
    end 
end