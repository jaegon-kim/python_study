import numpy as np 
inputs = np.array([[0.1, 0.2, 0.3, 0.4],
                   [2., 3., 4., 5.,],
                   [10., 11., 12., 13.],
                   [15., 16., 17., 18.]], dtype = np.float32) * 0.01

targets = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 1]], dtype=np.float32)

np.random.seed(220209)
W = np.random.randn(3, 4)
B = np.random.randn(3, 1)

learning_rate = 0.25

def forward(input, target, W, B):
    X = np.transpose(input, (1, 0))
    G = np.dot(W, X) + B
    G_T = np.transpose(G, (1, 0))
    S = np.exp(G_T) / np.sum(np.exp(G_T), axis = 1, keepdims=True)
    losses = np.sum(-target * np.log(S))
    return G, S, S, losses

def backward(G, S, input, target):
    dL_dG = S - target
    dL_dG = np.transpose(dL_dG, (1, 0))
    dG_dW = input 
    dL_dW = np.dot(dL_dG, dG_dW)
    dL_dB = np.sum(dL_dG, axis = 1, keepdims=True)
    dL_dB = np.sum(dL_dG, axis=1, keepdims=True)
    return dL_dW, dL_dB

_, _, pred, losses = forward(inputs, targets, W, B)
print('before pred: ', pred)
print('before loss: ', losses )

for i in range(1000):
    G1, S, pred, losses = forward(inputs, targets, W, B)
    dL_dW, dL_dB = backward(G1, S, inputs, targets)
    W = W + -1 * learning_rate * dL_dW
    B = B + -1 * learning_rate * dL_dB

_, _, pred, losses = forward(inputs, targets, W, B)
print('after pred: ', pred)
print('after loss : ', losses )

