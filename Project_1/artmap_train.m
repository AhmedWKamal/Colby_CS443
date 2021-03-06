function [C, w_code, w_out] = artmap_train(data_x, data_y, n_classes, verbose, show_plot, varargin)
%%artmap_train default ARTMAP training implementation of Fuzzy ARTMAP classifer with winner-take-all coding units
%
% Parameters:
%%%%%%%%%%%%%%%%%%%%
% data_x: matrix. size=(#dimensions (M), #samples (N)). Data samples normalized in the range [0,1].
% data_y: matrix of ints. size=(1, N). Classes represented as ints 1, 2..., #classes.
% n_classes: int. Number of classes in the dataset. This is a parameter because not all class values may be in the
%   training set. For example, our training set may have 4 classes, and the test set has 5 (one missing).
% verbose: boolean. If false, suppresses ALL print outs.
% show_plot: boolean. If true, show and update the category box plot during each iteration of the epoch
% varargin: cell array. variable length optional parameters.
%
% Returns:
%%%%%%%%%%%%%%%%%%%%
% C: int. Number of committed cells in the coding layer (y cells).
% w_code: array. size=(2*M, C_max). Input-to-coding-layer adaptive weights.
%   This tells us how to activate committed coding cells based on a new input pattern.
% w_out: array. size=(C_max, n_classes). Coding-layer-to-output-class-layer adaptive weights.
%   This tells us which output class each committed coding cell is associated with (which class it tends to predict).
%
% NOTE: We need both sets of learned weights w_code, w_out to form a prediction for a test input, which is why we're
% returning them.
% NOTE: C tells us which weights in w_code and w_out are currently used/relevant.

% Set parameters
%
% Coding layer y choice parameter ("tie breaker" for activation values). (0, 1)
alpha = 0.01;
% Learning rate. [0, 1]. 1 means fast one-shot learning
beta = 1;
% Matching tracking update rate. (-1, 1)
e = -0.001;
% Baseline vigilance / matching criterion. [0, 1]. 0 maximizes code compression.
p_base = 0;
% Number of training epochs. We only need 1 when beta=1
n_epochs = 1;
% Max number of commitable coding cells. C_max start uncommitted.
C_max = 20;
M = size(data_x,1);

%Initialize weights
w_code = ones(2*M,C_max);
w_out = zeros(C_max,n_classes);

%Set number of coded weights to 0
C = 0;
% complement code
data_x = complementCode(data_x);

%Commit the first node
[C, w_code, w_out] = addCommittedNode(C,data_x(:,1),data_y(:,1),w_code,w_out);

%For every epoch
for ep = 1:n_epochs
  %For every data point
  for i = 2:size(data_x,2)-1
    %Reset the vigilance
    p = p_base;
    %Compute net_in via choiceByDifference
    net_in = choiceByDifference(data_x(:,i),w_code,C,alpha,M);
    %Compute net_act by linear_thresholding of net_in
    %Returns the indices of the valid w_j's, in order
    [~, sorted_inds] = possibleMatchInds(net_in,alpha,M);
    net_act = net_in(sorted_inds);
   
    
    %For every valid weight
    for j = 1:size(sorted_inds, 1)
      %If the potential match passes the vigilance test
      if sum(min(w_code(:,sorted_inds(j)),(data_x(:,i))),"all") > (M * p)
        %And is of the same class as the data point being considered
        if find(w_out(sorted_inds(j), :)) == data_y(i)
          %Update the weight
          w_code = updateWts(beta, data_x(:,i), w_code, j);
          
          break;
          %But is of a different class
        else
          %Increase the vigilance
          p = matchTracking(data_x(:,i), w_code, sorted_inds(j), M, e);
          %Continue search cycle
        end
      end
      %Exhausted all nodes without finding a match
      if j == size(sorted_inds, 1)
        %Commit a new node
        [C, w_code, w_out] = addCommittedNode(C,data_x(:,i),data_y(:,i),w_code,w_out);
      end
    end % ART search cycle
    if show_plot
      plotCategoryBoxes(data_x, data_y, i, C, w_code, w_out, "train");
    end
  end % training sample 2->N loop
end

end