import numpy as np
import scipy
from scipy import integrate
import numpy, scipy.io

def generate_orth(n, m):
    X = np.random.normal(0,1,(n,m))
    if n < m:
        X = X.T
    Q = scipy.linalg.orth(X)
    if n < m:
        Q = Q.T
    return Q

#parameters
numSchz = 40
numHlth = 40
numPatients = numSchz + numHlth
numTR = 100
numVoxel = 10
k = 10
Srange = 1
Wrange = 1.1
Scovscale = 1
rhoscale = 25

#file names
relative_address = 'Generated Simple Data/'
Sc_file_name = 'Sc_data.npy'
Sh_file_name = 'Sh_data.npy'
Wc_file_name = 'Wc_data.npy'
Wh_file_name = 'Wh_data.npy'
Xc_file_name = 'Xc_data.npy'
Xh_file_name = 'Xh_data.npy'

#initialization
Sc = np.zeros([k, numTR])
Sh = np.zeros([k, numTR])
covScVals = np.ones(k) * Scovscale
covShVals = np.ones(k) * Scovscale
Wc = np.zeros([numSchz, numVoxel, k])
Wh = np.zeros([numHlth, numVoxel, k])
Xc = np.zeros([numSchz, numVoxel, numTR])
Xh = np.zeros([numHlth, numVoxel, numTR])

#generating S
meanSc = np.zeros(k)
covSc = covScVals * np.eye(k)
meanSh = np.zeros(k)
covSh = covShVals * np.eye(k)
for i in range(numTR):
    Sc[:, i] = np.random.multivariate_normal(meanSc, covSc, 1) * Srange
    Sh[:, i] = np.random.multivariate_normal(meanSh, covSh, 1) * Srange

#generating W
for i in range(numSchz):
    Wc[i] = generate_orth(numVoxel, k) * Wrange
    # Wc[i] = np.eye(k) * Wrange
for i in range(numHlth):
    Wh[i] = generate_orth(numVoxel, k) * Wrange * 10

#generating X
for i in range(numSchz):
    for t in range(numTR):
        Xc[i, :, t] = np.dot(Wc[i], Sc[:, t])

for i in range(numHlth):
    for t in range(numTR):
        Xh[i, :, t] = np.dot(Wc[i], Sc[:, t]) * 10

#save generated data into files
np.save(relative_address + Sc_file_name, Sc)
np.save(relative_address + Sh_file_name, Sh)
np.save(relative_address + Wc_file_name, Wc)
np.save(relative_address + Wh_file_name, Wh)
np.save(relative_address + Xc_file_name, Xc)
np.save(relative_address + Xh_file_name, Xh)