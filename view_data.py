import numpy as np

import matplotlib.pyplot as plt

data1 = np.load('/Users/farnazkohankhaki/PycharmProjects/Use_SRM/schyz1/srm/10feat/rand0/srm10_WS.npz')
data1S = data1['S']
print(data1S.shape)
data1W = data1['W']
data2 = np.load('/Users/farnazkohankhaki/PycharmProjects/Use_SRM/schyz2/srm/10feat/rand0/srm10_WS.npz')
data2S = data2['S']
data2W = data2['W']
dataRW = np.load('/Users/farnazkohankhaki/PycharmProjects/SRM_Data/Generated Simple Data/Wh_data.npy')
dataRS = np.load('/Users/farnazkohankhaki/PycharmProjects/SRM_Data/Generated Simple Data/Sh_data.npy')


dataSS = sum(data1S)
plt.plot(dataSS)
plt.show()



dataSS2 = sum(data2S)
plt.plot(dataSS2)
plt.show()

print("hi")