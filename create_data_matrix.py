# This script creates a single data file using the schizophrenia FMRI data
# with a matrix structure supported by the SRM model.

import numpy as np

input_matrix1 = np.load("Generated Simple Data/Xc_data.npy")
input_matrix2 = np.load("Generated Simple Data/Xh_data.npy")

data1 = []
data2 = []

for subject in range(input_matrix1.shape[0]):
    X = input_matrix1[subject]
    data1.append(X)

for subject in range(input_matrix2.shape[0]):
    X = input_matrix2[subject]
    data2.append(X)

np.savez("/Users/farnazkohankhaki/PycharmProjects/Use_SRM/data1", data=data1)
np.savez("/Users/farnazkohankhaki/PycharmProjects/Use_SRM/data2", data=data2)

np_Array = np.load("/Users/farnazkohankhaki/PycharmProjects/Use_SRM/data1.npz")
print(np_Array.files)

np_Array = np.load("/Users/farnazkohankhaki/PycharmProjects/Use_SRM/data2.npz")
print(np_Array.files)
