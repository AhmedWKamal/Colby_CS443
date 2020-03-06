function c_pred = fuzzy_art_predict(C, w_code, data, verbose, varargin)
%%Fuzzy ART Predict: Return the index of the coding cell that activates the most to each data sample.
%
% Parameters:
%%%%%%%%%%%%%%%%%%%%
% C: int. Number of committed cells in the coding layer (y cells).
% w_code: matrix. size=(2*M, C_max) Learned input-to-coding-layer adaptive weights.
% data: matrix. size=(#dimensions (M) x #samples (N)). Test Dataset. Values normalized in the range [0,1].
% verbose: boolean. If false, suppresses ALL print outs.
% varargin: cell array. variable length optional parameters.
%
% Returns:
%%%%%%%%%%%%%%%%%%%%
% c_pred. matrix. size=(# test samples, 1). Column vector contains the index of the activated coding layer cell
% in response to each test sample.


% Set parameters
%
% Coding layer y choice parameter. (0, Inf)
alpha = 0.01;

% Override default settings/parameters
for arg = 1:2:length(varargin)
  switch varargin{arg}
    case 'alpha'
      alpha = varargin{arg+1};
  end
end

M = size(data_x,1);

% complement code
data = complementCode(data);

%Commit the first node
[C, w_code] = addCommittedNode(C,data(:,1),w_code);

c_pred = zeros(n_classes, size(data, 2));
%For every epoch
for e = 1:n_epochs
  %For every data point
  for i = 1:size(data, 2)
    %Compute net_in via choiceByDifference
    net_in = choiceByWeber(data(:, i), w_code, C, alpha, M);
    %Compute net_act by linear_thresholding of net_in
    %Returns the indices of the valid w_j's, in order
    [~, sorted_inds] = sort(net_in, "descend"); %possibleMatchInds(net_in, alpha, M);
    c_pred(:, i) = w_code(sorted_inds(1), :);
    % The predicted class of the current test sample class is the index of the
    %coding-to-output weight vector coming from the most active coding cell that is nonzero.
    if show_plot
      plotCategoryBoxes(data, i, C, w_code, "test", c_pred);
    end
  end % training sample 2->N loop
end
end