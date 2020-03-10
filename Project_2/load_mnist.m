function [mnist_data, mnist_y] = load_mnist(mnist_path, which_set, num_exemplars, num_classes, do_plot)
  %%load_mnist Load, preprocess, reshape a subset or all of EITHER the MNIST digit training or test set.
  %
  % Parameters:
  %%%%%%%%%%%%%%%%%%%%
  % mnist_path: str. Relative path to MNIST dataset.
  % which_set: str. Either 'train' or 'test'. If 'train', load in MNIST training set. Otherwise, load test set.
  % num_exemplars: int. How many digits should select we from each class from the dataset? For example, if num_exemplars
  %   = 5, load the 1st 5 images of 0s in the dataset and also the 1st 5 1s, also the 1st 5 2s...etc. Note that order of
  %   digits in MNIST is NOT contiguous...i.e. not all the 0s are next to each other.
  % num_classes: int. How many digit classes are we going to load? For example if = 10, load digits of all types (0-9).
  %   If = 3, only load 0s, 1s, and 2s. You can assume for simplicity that selections are made in-order starting from 0.
  % do_plot: boolean. If true, make a square grid plot showing images of all the MNIST digits that you load in. Maybe
  %   for pratical reasons, cap the number of samples shown to something like 100.
  %
  % Returns:
  %%%%%%%%%%%%%%%%%%%%
  % mnist_data. matrix of doubles. size=(M (num_features), num_exemplars*num_classes).
  %   Each image normalized to the range [0-1] inclusive in double format.
  %   NOTE: The raw data is in UINT8 format, but each sample is already normalized [0, 255].
  % mnist_y. matrix of doubles. size=(num_exemplars*num_classes, 1).
  %   Int-codes of MNIST class for each sample. Codes should be ints in the range (inclusive) [0, 9] and literally mean
  %   which digit is shown in the image.
  
  if strcmp(which_set, 'train')
    mnist_data = load(mnist_path + "mnist_train.mat");
    mnist_data = mnist_data.data;
    mnist_y = load(mnist_path + "mnist_train_labels.mat");
    mnist_y = mnist_y.y;
  else
    mnist_data = load(mnist_path + "mnist_test.mat");
    mnist_data = mnist_data.data;
    mnist_y = load(mnist_path + "mnist_test_labels.mat");
    mnist_y = mnist_y.y;
  end
  
  
  mnist_data_ret = [];
  mnist_y_ret = [];
  
  %Create 
  for i = 1:num_classes
      logical_indices = (mnist_y(:) == i-1);
      temp_x = mnist_data(logical_indices,:);
      temp_y = mnist_y(logical_indices);
      mnist_data_ret = [mnist_data_ret; temp_x(1:num_exemplars,:)];
      mnist_y_ret = [mnist_y_ret;temp_y(1:num_exemplars)'];
  end
  
  mnist_data = mnist_data_ret';
  disp(size(mnist_data))
  mnist_y = mnist_y_ret;
  for i = 1:num_classes
      for j = 1:min(10,num_exemplars)
          disp(num2str((i-1)*num_exemplars+j)+"-"+num2str(i+num_classes*(j)))
          subplot(num_exemplars,num_classes,(i-1)*num_exemplars+j)
          img_mat = reshape(mnist_data(:,i+num_classes*(j)),28,28);
          imshow(img_mat')
      end
  end
  mnist_data = mnist_data/255;
  
  
end

