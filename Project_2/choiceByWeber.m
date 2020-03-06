function Tj = choiceByWeber(curr_A, w_code, alpha, C)
%%choiceByWeber Choice-by-Weber coding layer input match function
%
% Parameters
%%%%%%%%%%%%
% curr_A: Current input
% w_code: Coding layer wts for committed nodes only!
% alpha: Choice parameter

Tj = [];
for i = 1:C
  C = C +1;
  a = norm(min(w_code(:,i),curr_A),1);
  b = (alpha + norm(w_code(:,i), 1));
  Tj = a / b ;
end
end