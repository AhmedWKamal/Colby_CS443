function yh_pred = artmap_test_wta(C, w_code, w_out, data_x, data_y, n_classes, verbose, show_plot, varargin)
%%ARTMAP Default ARTMAP implementation of the Fuzzy ARTMAP classifer with winner-take-all testing.
%
% Parameters:
%%%%%%%%%%%%%%%%%%%%
% C: int. Number of committed cells in the coding layer (y cells).
% w_code: matrix. size=(2*M, C_max) Learned input-to-coding-layer adaptive weights.
% w_out: matrix. size=(C_max, n_classes). Learned coding-layer-to-output-class-layer adaptive weights.
% data_x: matrix. size=(#dimensions (M) x #samples (N)). Test Dataset. Values normalized in the range [0,1].
% data_y: matrix of ints. size=(1, N). Classes represented as ints 1, 2..., #classes.
%   NOTE: Only used for plotting purposes. Should not be used in algorithm!
% n_classes: int. Number of classes in the dataset. This is a parameter because not all class values may be in the
%   test set. For example, our test set may have 4 classes, and the training set has 5 (one missing).
% verbose: boolean. If false, suppresses ALL print outs.
% show_plot: boolean. If true, show the category box plot during each iteration of the epoch.
% varargin: cell array. variable length optional parameters.
%
% Returns:
%%%%%%%%%%%%%%%%%%%%
% yh_pred. matrix. size=(n_classes, # test samples). Each column vector contains the prediction for the corresponding
% test sample. In fast learning this is a matrix of one-hot coded vectors: every vector contains a 1 in the predicted
% class, 0s elsewhere.


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

M = size(data_x,1);

% complement code
data_x = complementCode(data_x);

%Commit the first node
[C, w_code, w_out] = addCommittedNode(C,data_x(:,1),data_y(:,1),w_code,w_out);

yh_pred = zeros(n_classes, size(data_y, 2));
%For every epoch
for e = 1:n_epochs
  %For every data point
  for i = 1:size(data_x, 2)
    %Reset the vigilance
    p = p_base;
    %Compute net_in via choiceByDifference
    net_in = choiceByDifference(data_x(:, i), w_code, C, alpha, M);
    %Compute net_act by linear_thresholding of net_in
    %Returns the indices of the valid w_j's, in order
    [~, sorted_inds] = sort(net_in, "descend"); %possibleMatchInds(net_in, alpha, M);
    yh_pred(:, i) = w_out(sorted_inds(1), :);
    % The predicted class of the current test sample class is the index of the
    %coding-to-output weight vector coming from the most active coding cell that is nonzero.
    if show_plot
      plotCategoryBoxes(data_x, data_y, i, C, w_code, w_out, "test", yh_pred);  
    end
  end % training sample 2->N loop
end
end