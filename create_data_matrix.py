# This script creates a single data file using the schizophrenia FMRI data
# with a matrix structure supported by the SRM model.

import numpy as np

input_matrix = np.load("../schizophrenia/Xh_data.npy")

data = []

for subject in range(input_matrix.shape[0]):
    X = input_matrix[subject]
    data.append(X)

np.savez("data", data=data)

np_Array = np.load("data.npz")
print(np_Array.files)
