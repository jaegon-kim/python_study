import numpy as np 
from numpy import genfromtxt
import matplotlib.pyplot as plt 

data = genfromtxt('weight_height.csv', delimiter=',', skip_header = 1)
inputs = data[:, 0].reshape(-1, 1) * 0.01
targets = data[:, 1].reshape(-1, 1) * 0.01

np.random.seed(220120)

W1 = np.random.randn(8, 1)
B1 = np.random.randn(8, 1)
W2 = np.random.randn(4, 8)
B2 = np.random.randn(4, 1)
W3 = np.random.randn(1, 4)
B3 = np.random.randn(1, 1)

learning_rate = 0.1

def forward(input, target, W1, B1, W2, B2, W3, B3):
    X = np.transpose(input, (1, 0))
    G1 = np.dot(W1, X) + B1
    R1 = np.maximum(0, G1)
    G2 = np.dot(W2, R1) + B2
    R2 = np.maximum(0, G2)
    G3 = np.dot(W3, R2) + B3
    pred = np.transpose(G3, (1, 0))
    loss = np.mean(np.power(pred-target, 2))
    return G1, R1, G2, R2, G3, pred, loss

def loss_gradient(input, target, W1, B1, W2, B2, W3, B3):
    G1, R1, G2, R2, G3, _, _ = forward(input, target, W1, B1, W2, B2, W3, B3)
    target = np.transpose(target, (1, 0))
    dL_dG3 = 2 * (G3-target) / len(target[0])
    dG3_dR2 = np.transpose(W3, (1, 0))
    dG3_dW3 = np.transpose(R2, (1, 0))
    dG3_dB3 = np.ones_like(B3)
    dR2_dG2 = np.ones_like(G2)
    dG2_dR1 = np.transpose(W2, (1,0))
    dG2_dW2 = np.transpose(R1, (1,0))
    dG2_dB2 = np.ones_like(B2)
    dR1_dG1 = np.ones_like(G1)
    dG1_dW1 = input
    dG1_dB1 = np.ones_like(B3)

    dL_dW3 = np.dot(dL_dG3, dG3_dW3)
    dL_dB3 = np.sum(dL_dG3, keepdims=True) * dG3_dB3

    dL_dG2 = np.dot(dG3_dR2, dL_dG3) * dR2_dG2
    dL_dW2 = np.dot(dL_dG2, dG2_dW2)
    dL_dB2 = np.sum(dL_dG2, axis = 1, keepdims=True) * dG2_dB2

    dL_dG1 = np.dot(dG2_dR1, dL_dG2) * dR1_dG1
    dL_dW1 = np.dot(dL_dG1, dG1_dW1)
    dL_dB1 = np.sum(dL_dG1, axis = 1, keepdims=True) * dG1_dB1

    return dL_dW1, dL_dB1, dL_dW2, dL_dB2, dL_dW3, dL_dB3 

_, _, _, _, _, pred, loss = forward(inputs, targets, W1, B1, W2, B2, W3, B3)
print('before loss: ', loss)

for i in range(2000):
    dL_dW1, dL_dB1, dL_dW2, dL_dB2, dL_dW3, dL_dB3 = loss_gradient(inputs, targets, W1, B1, W2, B2, W3, B3)
    W1 = W1 + -1*learning_rate*dL_dW1
    B1 = B1 + -1*learning_rate*dL_dB1
    W2 = W2 + -1*learning_rate*dL_dW2
    B2 = B2 + -1*learning_rate*dL_dB2
    W3 = W3 + -1*learning_rate*dL_dW3
    B3 = B3 + -1*learning_rate*dL_dB3

    _, _, _, _, _, pred, loss = forward(inputs, targets, W1, B1, W2, B2, W3, B3)
print('after loss: ', loss)

plt.plot(inputs*100, targets*100, 'ro', label='target')
plt.plot(inputs*100, pred*100, 'bo', label='pred')
plt.xlabel('weight', size=15)
plt.ylabel('height', size=15)
plt.legend()
plt.show()


