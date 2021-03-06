function [mnist_test_y, code_inds, C] =  run_fuzzy_art_mnist(mnist_path, sets, ...
    num_exemplars, num_classes, noisify_test, erase_test, plot_wts, plot_recall, varargin)
  %%run_fuzzy_art_mnist Trains fuzzy ART on the MNIST dataset and recover "memories" prompted by the test set using the
  %%Fuzzy ART predict function. A memory refers to the weights for the winning code unit for each test data sample.
  %
  % Parameters:
  %%%%%%%%%%%%%%%%%%%%
  % mnist_path: str. Relative path to MNIST dataset.
  % sets: cell array of strings. size=(2, 1). Entries contain either 'train' or 'test'. 1st entry is which MNIST set is
  %   used during training. 2nd entry is MNIST set used during prediction. For example, {'test', 'test'} would mean use
  %   the test set for both training and prediction. If 'train', load in MNIST training set. Otherwise, load test set.
  % num_exemplars: matrix of ints. size=(2, 1).  Each number means how many digits should select we from each class from
  %   the dataset? For example, if num_exemplars = 5, load the 1st 5 images of 0s in the dataset and also the 1st 5 1s,
  %   also the 1st 5 2s...etc. 
  %   1st entry is how many exemplars should be loaded for training. 2nd entry is for how many load in prediction
  %   (test). For example, [5, 1] means load 5 samples of each digit from MNIST during training and 1 per digit during
  %   prediction/testing.
  % num_classes: int. How many digit classes are we going to load? For example if = 10, load digits of all types (0-9).
  %   If = 3, only load 0s, 1s, and 2s. You can assume for simplicity that selections are made in-order starting from 0.
  % noisify_test: boolean. Do we add noise to the test images? Ignore/set to false until the notebook instructions state
  %   otherwise.
  % erase_test: boolean. Do we erase part of each test image? Ignore/set to false until the notebook instructions state
  %   otherwise.  
  % plot_wts: boolean. If true, make a square grid plot showing images of all the learned weights of each of the committed
  % coding units. Reshaping to 28x28 pixels will be necessary.
  % plot_recall: boolean. If true, make a 2 column figure.
  %   Each row in left column shows each input test set image.
  %   Each row in right column shows the corresponding weight (memory) of the coding unit that was most active when you
  %   presented the test image to Fuzzy ART (determined in predict function).
  % varargin: cell array. size=variable. List of strings and values to pass along to fuzzy_art_train (and be parsed
  % there). Holds things like hyperparameters (e.g. p, alpha, beta, num_epochs, etc.).
  %
  % Returns:
  %%%%%%%%%%%%%%%%%%%%
  % mnist_test_y. matrix of doubles. size=(num_test_exemplars*num_classes, 1).
  %   Int-codes of MNIST class for each test sample. Codes should be ints in the range (inclusive) [0, 9] and literally
  % mean which digit is shown in the image.
  % code_inds. matrix of ints. size=(num_test_exemplars*num_classes, 1).
  %   Indices of the max active coding units to each test image.
  % C. int. Number of committed coding layer units.
  %
  % TODO: (also see notebook)
  % - Load MNIST train and test sets.
  % - Train Fuzzy ART on training set.
  % - Possibly visualize the weights.
  % - Predict the categories of the test set images.
  % - Possibly visualize the weights of the most active coding units during prediction to the test images.
  
  %disp("Loading data...")
  [mnist_train_data, ~] = load_mnist(mnist_path, sets{1,1}, num_exemplars(1,1), num_classes, false);
  [mnist_test_data, mnist_test_y] = load_mnist(mnist_path, sets{2,1}, num_exemplars(2,1), num_classes, false);
  %disp("Training...")
  [C, w_code]= fuzzy_art_train(mnist_train_data, false, varargin{:});
  if noisify_test
      %disp("Adding noise to test data...")
      f = 0.84;
      num_corrupted_pixels = ceil(f*size(mnist_test_data,1));
      indices_to_corrupt = randperm(size(mnist_test_data,1),num_corrupted_pixels);
      for i = 1:size(mnist_test_data,2)
          I = ceil(num_corrupted_pixels/2);
          %Set half to 0
          mnist_test_data(indices_to_corrupt(1:I),:) = 0;
          %Set half to 1
          mnist_test_data(indices_to_corrupt(I:end),:) = 1;
      end
  end
  
  if erase_test
      disp("Erasing part of test data...")
      f = 0.5;
      cols_erased = 1:ceil(f*28);
      for i = 1:size(mnist_test_data,2)
          mat = reshape(mnist_test_data(:,i),28,28)';
          mat(:,cols_erased) = 0;
          mnist_test_data(:,i) = reshape(mat',784,1);
      end
  end
  %disp("Using net to predict...")
  code_inds = fuzzy_art_predict(C, w_code, mnist_test_data, false);
  if plot_wts
      %disp("Plotting weights...")
      figure();
      square_grid_size = ceil(sqrt(C));
      for i = 1:C
          subplot(square_grid_size,square_grid_size,i);
          img = uint8(w_code(1:784,i)*255);
          img_mat = reshape(img,28,28);
          imshow(img_mat');
      end
  end
  if plot_recall
      %disp("Plotting recall...")
      figure();
      for i = 1:size(mnist_test_data,2)
          %Show the test data
          subplot(size(mnist_test_data,2),2,2*i-1);
          img = mnist_test_data(:,i);
          img_mat = reshape(img,28,28);
          imshow(img_mat');
          %Show the recalled memory
          subplot(size(mnist_test_data,2),2,2*i);
          img = uint8(w_code(1:784,code_inds(i))*255);
          img_mat = reshape(img,28,28);
          imshow(img_mat');
      end
  end

end

