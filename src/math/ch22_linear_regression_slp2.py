import numpy as np 
import matplotlib.pyplot as plt 
from numpy import genfromtxt 

np.random.seed(220112)

data = genfromtxt('weight_height.csv', delimiter=',', skip_header=1)
'''
input = data[:, 0]
target = data[:, 1]

input = input.reshape(-1, 1)
target = target.reshape(-1, 1)

plt.plot(input, target, 'ro')
plt.xlabel('weight', size=15)
plt.ylabel('height', size=15)
plt.show()
'''

input = data[:, 0] * 0.1
target = data[:, 1] * 0.1

input = input.reshape(-1, 1) # 가로열 배열을 세로열 배열로 reshape.
input2 = np.hstack((np.power(input, 2), input))
target = target.reshape(-1, 1) # 가로열 배열을 세로열 배열로 reshape

W = np.random.randn(1, 1)
B = np.random.randn(1, 1)

W2 = np.random.randn(2, 1)
B2 = np.random.randn(1, 1)

learning_rate = 0.001
learning_rate2 = 0.00005

def linear_forward(X, Y, W, B):
    pred = np.dot(X, W) + B
    loss = np.mean(np.power(Y - pred, 2))
    return pred, loss

def loss_gradient(X, Y, W, B):
    XWB = np.dot(X, W) + B
    dL_dg = 2 * (XWB - Y)
    dg_dW = np.transpose(X, (1, 0))
    dL_dW = np.dot(dg_dW, dL_dg)
    dL_dB = np.sum(dL_dg, axis=0)
    return dL_dW, dL_dB

pred, loss = linear_forward(input, target, W, B)
pred2, loss2 = linear_forward(input2, target, W2, B2)
print('loss: ', loss)
print('loss2: ', loss2)

for i in range(100):
    dL_dW, dL_dB = loss_gradient(input, target, W, B)
    W = W + -1 * learning_rate * dL_dW
    B = B + -1 * learning_rate * dL_dB

for i in range(100):
    dL_dW, dL_dB = loss_gradient(input2, target, W2, B2)
    W2 = W2 + -1 * learning_rate2 * dL_dW
    B2 = B2 + -1 * learning_rate2 * dL_dB

pred, loss = linear_forward(input, target, W, B)
pred2, loss2 = linear_forward(input2, target, W2, B2)
print('loss: ', loss)
print('loss2: ', loss2)

print('W2: ', W2)
print('B2: ', B2)

plt.plot(input, target, 'ro', label='target')
plt.plot(input, pred, 'b', label='pred')
plt.plot(input, pred2, 'g', label='pred2')
plt.xlabel('weight', size=15)
plt.ylabel('height', size=15)
plt.legend()
plt.show()
