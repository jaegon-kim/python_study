import numpy as np 
import matplotlib.pyplot as plt 
from numpy import genfromtxt

data = genfromtxt('2017_seoul_temperature-2C.csv', delimiter=',', skip_header=1)
days = data[:, 0]
temp = data[:, 1:]

alpha_05 = 0.5
alpha_09 = 0.9
alpha_099 = 0.99

EWMA_05 = []
EWMA_09 = []
EWMA_099 = []

for i in range(len(days)):
    if i == 0:
        Y = temp[0]
    else:
        Y = alpha_05 * Y + (1 - alpha_05) * temp[i]
    EWMA_05.append(Y[0])

for i in range(len(days)):
    if i == 0:
        Y = temp[0]
    else:
        Y = alpha_09 * Y + (1 - alpha_09) * temp[i]
    EWMA_09.append(Y[0])

for i in range(len(days)):
    if i == 0:
        Y = temp[0]
    else:
        Y = alpha_099 * Y + (1 - alpha_099) * temp[i]
    EWMA_099.append(Y[0])

means = np.repeat(np.mean(temp), len(temp))
moving_avg = np.convolve(temp.flatten(), np.ones(30), 'same')/30
weighted_moving_avg = np.convolve(temp.flatten(), np.arange(1, 31), 'same')/np.sum(np.arange(1, 31))

plt.plot(days, temp, 'ro', markersize=3, label='original')
plt.plot(days, means, color='#FF6B33', markersize=3, label='mean')
plt.plot(days, moving_avg, color='#C205B9', markersize=3, label='moving average')
plt.plot(days, weighted_moving_avg, color='#39C205', markersize=3, label='moving average')

plt.plot(days, EWMA_05, color='#D1F529', markersize=3, label='EWMA 0.5')
plt.plot(days, EWMA_09, color='#000000', markersize=3, label='EWMA 0.9')
plt.plot(days, EWMA_099, color='#253F85', markersize=3, label='EWMA 0.99')


plt.xlabel('days')
plt.ylabel('temperature')
plt.legend()

plt.show()