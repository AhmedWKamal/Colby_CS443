function Tj = choiceByWeber(curr_A, w_code, alpha)
  %%choiceByWeber Choice-by-Weber coding layer input match function
  %
  % Parameters
  %%%%%%%%%%%%
  % curr_A: Current input
  % w_code: Coding layer wts for committed nodes only!
  % alpha: Choice parameter
  
  Tj = [];
  for i = 1:C
      a = sum(min(w_code(:,i),curr_A),"all");
      b = (alpha + min(w_code(:,i));
      Tj = [Tj; a/b];
    end
end