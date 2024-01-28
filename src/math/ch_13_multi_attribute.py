import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

np.random.seed(220103)

input = np.array([[0, 0],
                  [1., 0],
                  [0, 1.],
                  [1., 1.]])

target = np.array([[1.],
                   [3.],
                   [4.],
                   [6.]])

# Original : Y = 2 * X1 + 3 * X2 + 1

W = np.random.randn(2, 1) 
B = np.random.randn(1, 1)

print('W: ', W)
print('B: ', B)

learning_rate = 0.005

def linear_forward(X, Y, W, B):
    pred = np.dot(X, W) + B
    loss = np.mean(np.power(Y - pred, 2))
    return pred, loss

def loss_gradient(X, Y, W, B):
    # ∂L(g(X, W, B)) / ∂g(X, W, B)
    dL_dg = 2 *(np.dot(X, W) + B - Y)

    # ∂g(X, W, B) / ∂W
    dg_dW = np.transpose(X, (1, 0))

    # ∂L(g(X, W, B)) / ∂W
    dL_dW = np.dot(dg_dW, dL_dg)

    # ∂L(g(X, W, B)) / dB
    dL_dB = np.sum(dL_dg, axis = 0)

    return dL_dW, dL_dB

pred, loss = linear_forward(input, target, W, B)
print('pred: ', pred)
print('loss: ', loss)


dL_dW, dL_dB = loss_gradient(input, target, W, B)
print('before weight: ', W)
print('before bias  : ', B)

for i in range(500):
    dL_dW, dL_dB = loss_gradient(input, target, W, B)
    W = W + -1 * learning_rate * dL_dW
    B = B + -1 * learning_rate * dL_dB

print('after weight: ', W)
print('after bias  : ', B)


pred, loss = linear_forward(input, target, W, B)
print('pred: ', pred)
print('loss: ', loss)