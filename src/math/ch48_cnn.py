import numpy as np 

a = np.array([[1., 0., 0.],
              [0., 1., 0.],
              [0., 0., 1.]])

b = np.array([[1., 0., 1.],
              [0., 0., 0.],
              [1., 0., 1.]])

c = np.array([[0., 1., 0.],
              [1., 0., 1.],
              [0., 1., 0.]])

filter = np.array([[1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]])

out_a = np.sum(a * filter)
out_b = np.sum(b * filter)
out_c = np.sum(c * filter)
print(out_a)
print(out_b)
print(out_c)


import matplotlib.pyplot as plt 

fig, ax = plt.subplots(2, 3, figsize=(8, 5))

ax[0][0].imshow(a * filter, cmap='gray')
ax[0][1].imshow(b * filter, cmap='gray')
ax[0][2].imshow(c * filter, cmap='gray')
ax[1][1].imshow(filter, cmap='gray')
#plt.show()



