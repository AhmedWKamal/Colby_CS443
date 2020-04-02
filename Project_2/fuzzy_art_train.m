function [C, w_code] = fuzzy_art_train(data, verbose, varargin)
    %%fuzzy_art fuzzy ART unsupervised pattern learning algorithm
    %
    % Parameters:
    %%%%%%%%%%%%%%%%%%%%
    % data: matrix. size=(#dimensions (M), #samples (N)). Data samples normalized in the range [0,1].
    % verbose: boolean. If false, suppresses ALL print outs.
    % show_plot: boolean. If true, show and update the category box plot during each iteration of the epoch.
    %   Only applies to CIS dataset.
    % varargin: cell array. variable length optional parameters.
    %
    % Returns:
    %%%%%%%%%%%%%%%%%%%%
    % C: int. Number of committed cells in the coding layer (y cells).
    % w_code: array. size=(2*M, C_max). Input-to-coding-layer adaptive weights.
    %   This tells us how to activate committed coding cells based on a new input pattern.
    %
    % NOTE: We need both sets of learned weights w_code, w_out to form a prediction for a test input, which is why we're
    % returning them.
    % NOTE: C tells us which weights in w_code and w_out are currently used/relevant.

    % Set parameters
    %
    % Coding layer y choice parameter ("tie breaker" for activation values). (0, Inf)
    alpha = 0.01;
    % Learning rate. [0, 1]. 1 means fast one-shot learning
    beta = 1.0;

    % Baseline vigilance / matching criterion. [0, 1]. 0 maximizes code compression.
    p = 0.75;
    % Number of training epochs. We only need 1 when beta=1
    n_epochs = 1;

    % Max number of commitable coding cells. C_max start uncommitted.
    C_max = size(data, 2);
    show_plot = false;
    fcsr = false;
    % Override default settings/parameters
    for arg = 1:2:length(varargin)
        switch varargin{arg}
            case 'alpha'
                alpha = varargin{arg+1};
            case 'beta'
                beta = varargin{arg+1};
            case 'p'
                p = varargin{arg+1};
            case 'n_epochs'
                n_epochs = varargin{arg+1};
            case 'C_max'
                C_max = varargin{arg+1};
            case 'show_plot'
                show_plot = varargin{arg+1};
            case 'fast_commit_slow_recode'
                fcsr = varargin{arg+1};
        end
    end
    M = size(data,1);
    %Initialize weights
    w_code = ones(2*M,C_max);
    %Set number of coded weights to 0
    C = 0;
    % complement code
    data_x = complementCode(data);

    %Commit the first node
    [C, w_code] = addCommittedNode(C,data_x(:,1),w_code);
    if fcsr
        %For every epoch
        for ep = 1:n_epochs
            %For every data point
            for i = 2:size(data_x,2)
                %Compute net_in via Weber Laws
                Tj = choiceByWeber( data_x(:,i), w_code, alpha, C_max );
                %Returns the indices of the valid w_j's, in order
                [~, sorted_inds] = sort(Tj, "descend");
                %For every valid weight
                for j = 1:size(sorted_inds, 2) %sorted max to min
                    %If the potential match passes the vigilance 
                    test_val = norm(min(w_code(:,sorted_inds(j)),data_x(:,i)),1)/norm(data_x(:,i),1);
                    if test_val >= p %vigilance test
                        % If match is an uncommitted cell, commit it and break out
                        if sorted_inds(j) > C %match uncommited
                            C = C+1;
                            w_code = updateWts(1, data_x(:,i), w_code, sorted_inds(j));
                            break;
                        else
                            % Else match is with a committed cell, updates its wts break out
                            %Update the weight
                            w_code = updateWts(beta, data_x(:,i), w_code, sorted_inds(j));
                            break; 
                        end
                        %Continue search cycle
                    end
                end % END ART search cycle
            end % END processing data samples
            if show_plot == true
                plotCategoryBoxes(data_x, i, C, w_code);
            end
        end % training sample 2->N loop
    else
        %For every epoch
        for ep = 1:n_epochs
            %For every data point
            for i = 2:size(data_x,2)
                %Compute net_in via Weber Laws
                Tj = choiceByWeber( data_x(:,i), w_code, alpha, C_max );
                %Returns the indices of the valid w_j's, in order
                [~, sorted_inds] = sort(Tj, "descend");
                %For every valid weight
                for j = 1:size(sorted_inds, 2) %sorted max to min
                    %If the potential match passes the vigilance 
                    test_val = norm(min(w_code(:,sorted_inds(j)),data_x(:,i)),1)/norm(data_x(:,i),1);
                    if test_val >= p %vigilance test
                        % If match is an uncommitted cell, commit it and break out
                        if sorted_inds(j) > C %match uncommited
                            C = C+1;
                            w_code = updateWts(beta, data_x(:,i), w_code, sorted_inds(j));
                            break;
                        else
                            % Else match is with a committed cell, updates its wts break out
                            %Update the weight
                            w_code = updateWts(beta, data_x(:,i), w_code, sorted_inds(j));
                            break; 
                        end
                        %Continue search cycle
                    end
                end % END ART search cycle
            end % END processing data samples
            if show_plot == true
                figure();
                plotCategoryBoxes(data_x, i, C, w_code);
            end
        end % training sample 2->N loop
    end
end