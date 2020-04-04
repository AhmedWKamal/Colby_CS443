# 3g ---------------------------------------------------------------------------------------------------------
# create initial binary noise mask
ten_percent = int(n_rows*n_cols*0.1)
mask_list = (ten_percent * [-1]) + (((n_rows*n_cols)-ten_percent) * [1])
mask = np.array(mask_list)
mask_test = np.copy(mask)

# for each digit, add 10% noise to the image
for i in range(n_digits):
    np.random.shuffle(mask_test)
    mask_vis = np.reshape(mask_test, (n_rows, n_cols))
    data_vis[i] = data_vis[i]*mask_vis
    data_test[i] = data_test[i]*mask_test


# 3h ---------------------------------------------------------------------------------------------------------
# for each item, lose the bottom half of the image
noisy_digits_test = np.concatenate(noisy_digits_test[:, :int(n_rows*n_cols*0.5)], np.zeros(n_rows*n_cols-int(n_rows*n_cols*0.5)))
noisy_digits_vis = np.reshape(noisy_digits_test, (n_rows, n_cols))


# 4a ---------------------------------------------------------------------------------------------------------
n_digits = 10
n_rows  = 7
n_cols = 9

# Import, preprocess, and plot digits ultimately as ndarrays  

# load binary digit data
with open("data/digits.txt") as f:
    content = f.read()
    
digits = np.array([int(char) for line in content.split('\n') for char in line])
digits = np.where(digits==0, -1, digits)

# create dataframe digits for visualization and training
digits_vis = digits.reshape((n_digits, n_rows, n_cols))
digits_train = digits.reshape((n_digits, n_rows*n_cols))

# plot the digits
row_plot(digits_vis, 'Raw training digit images', cmap='gray')


# 4b ---------------------------------------------------------------------------------------------------------
# create initial binary noise mask
ten_percent = int(n_rows*n_cols*0.1)
mask_list = (ten_percent * [-1]) + (((n_rows*n_cols)-ten_percent) * [1])
mask = np.array(mask_list)
mask_test = np.copy(mask)

# copy over digits
noisy_digits_vis = np.copy(digits_vis)
noisy_digits_test = np.copy(digits_train)


# for each digit, add 10% noise to the image
for i in range(n_digits):
    np.random.shuffle(mask_test)
    mask_vis = np.reshape(mask_test, (n_rows, n_cols))
    noisy_digits_vis[i] = noisy_digits_vis[i]*mask_vis
    noisy_digits_test[i] = noisy_digits_test[i]*mask_test

row_plot(digits_vis, 'original images', cmap='gray')
row_plot(noisy_digits_vis, 'noisy images', cmap='gray')

