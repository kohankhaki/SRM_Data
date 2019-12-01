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
numVoxel = 20
k = 7
mucRange = 100
muhRange = 100
Scovscale = 15
rhoscale = 25

#file names
relative_address = 'Generated Data/'
Sc_file_name = 'Sc_data.npy'
Sh_file_name = 'Sh_data.npy'
Wc_file_name = 'Wc_data.npy'
Wh_file_name = 'Wh_data.npy'
Xc_file_name = 'Xc_data.npy'
Xh_file_name = 'Xh_data.npy'
Xchat_file_name = 'Xchat_data.npy'
Xhhat_file_name = 'Xhhat_data.npy'

#initialization
Sc = np.zeros([k, numTR])
Sh = np.zeros([k, numTR])
covScVals = np.random.rand(k) / Scovscale
covShVals = np.random.rand(k) / Scovscale
Wc = np.zeros([numSchz, numVoxel, k])
rhoc = np.random.rand(numSchz) / rhoscale
Wh = np.zeros([numHlth, numVoxel, k])
rhoh = np.random.rand(numHlth) / rhoscale
muc = np.zeros([numSchz, numVoxel, numTR])
Xc = np.zeros([numSchz, numVoxel, numTR])
Xchat = np.zeros([numSchz, numVoxel, numTR])
muh = np.zeros([numHlth, numVoxel, numTR])
Xh = np.zeros([numHlth, numVoxel, numTR])
Xhhat = np.zeros([numHlth, numVoxel, numTR])

#generating S
meanSc = np.zeros(k)
covSc = covScVals * np.eye(k)
meanSh = np.zeros(k)
covSh = covShVals * np.eye(k)
for i in range(numTR):
    Sc[:, i] = np.random.multivariate_normal(meanSc, covSc, 1)
    Sh[:, i] = np.random.multivariate_normal(meanSh, covSh, 1)

#generating W
for i in range(numSchz):
    Wc[i] = generate_orth(numVoxel, k)
for i in range(numHlth):
    Wh[i] = generate_orth(numVoxel, k)

#generating mu
muc = np.ones([numSchz, numVoxel, numTR]) * mucRange
muh = np.ones([numHlth, numVoxel, numTR]) * muhRange

#generating X
for i in range(numSchz):
    for t in range(numTR):
        Xc[i, :, t] = np.dot(Wc[i], Sc[:, t]) + muc[i, :, t]

for i in range(numHlth):
    for t in range(numTR):
        Xh[i, :, t] = np.dot(Wh[i], Sh[:, t]) + muh[i, :, t]

#generating Xhat
for i in range(numSchz):
    for t in range(numTR):
        mean = np.dot(Wc[i], Sc[:, t]) + muc[i, :, t]
        cov = rhoc[i] ** 2 * np.eye(numVoxel)
        Xchat[i, :, t] = np.random.multivariate_normal(mean, cov, 1)

for i in range(numHlth):
    for t in range(numTR):
        mean = np.dot(Wh[i], Sh[:, t]) + muh[i, :, t]
        cov = rhoh[i] ** 2 * np.eye(numVoxel)
        Xhhat[i, :, t] = np.random.multivariate_normal(mean, cov, 1)

#save generated data into files
np.save(relative_address + Sc_file_name, Sc)
np.save(relative_address + Sh_file_name, Sh)
np.save(relative_address + Wc_file_name, Wc)
np.save(relative_address + Wh_file_name, Wh)
np.save(relative_address + Xc_file_name, Xc)
np.save(relative_address + Xh_file_name, Xh)
np.save(relative_address + Xchat_file_name, Xchat)
np.save(relative_address + Xhhat_file_name, Xhhat)

