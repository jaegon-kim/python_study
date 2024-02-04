import numpy as np 

v = np.arange(6)
print('v: ', v, ' shape: ', v.shape)

r = v.reshape(1, -1)
print('r: ', r, ' shape: ', r.shape)


for i in range(3):
    print(i)

for i in reversed(range(3)):
    print(i)

a = np.array([0, 0, 0, 0])
softmax = np.exp(a) / np.sum(np.exp(a))
print(softmax)