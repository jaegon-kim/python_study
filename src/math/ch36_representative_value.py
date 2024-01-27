# https://toyourlight.tistory.com/36

import numpy as np
import matplotlib.pyplot as plt 
from numpy import genfromtxt

data = genfromtxt('weight_height.csv', delimiter=',', skip_header=1)

wrong_data = np.array([[70, 210]])
data = np.concatenate((data, wrong_data), axis=0)

weight = data[:, 0] # 모든 row에 대해서 column 0의 값
height = data[:, 1] # 모든 row에 대해서 column 1의 값

fig, ax = plt.subplots(1, 3)

ax[0].plot(weight, 'ro', markersize=2, label='weight')
ax[0].plot(height, 'bo', markersize=2, label='height')
ax[0].set_title('Original data', fontsize=10)
ax[0].legend()

Minmax_weight = (weight-weight.min()) / (weight.max() - weight.min())
Minmax_height = (height-height.min()) / (height.max() - height.min())

ax[1].plot(Minmax_weight, 'ro', markersize=2, label='weight')
ax[1].plot(Minmax_height, 'bo', markersize=2, label='height')
ax[1].set_title('Normalization', fontsize=10)
ax[1].legend()

Zscore_weight = (weight-weight.mean())/weight.std()
Zscore_height = (height-height.mean())/height.std()

ax[2].plot(Zscore_weight, 'ro', markersize=2, label='weight')
ax[2].plot(Zscore_height, 'bo', markersize=2, label='height')
ax[2].set_title('Standardization', fontsize=10)
ax[2].legend()

print('Zscore_weight mean', Zscore_weight.mean())
print('Zscore_weight std', Zscore_weight.std())

print('Zscore_height mean', Zscore_height.mean())
print('Zscore_height std', Zscore_height.std())


plt.show()