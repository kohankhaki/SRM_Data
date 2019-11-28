# This script creates a single data file using the schizophrenia FMRI data
# with a matrix structure supported by the SRM model.

import numpy as np

input_matrix = np.load("Generated Simple Data/Xc_data.npy")

data = []

for subject in range(input_matrix.shape[0]):
    X = input_matrix[subject]
    data.append(X)

np.savez("/Users/farnazkohankhaki/PycharmProjects/Use_SRM/data2", data=data)

np_Array = np.load("/Users/farnazkohankhaki/PycharmProjects/Use_SRM/data2.npz")
print(np_Array.files)
