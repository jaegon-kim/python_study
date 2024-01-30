import numpy as np

np.random.seed(220132)

inputs = np.array([[0., 0.],
                   [1., 0.],
                   [0., 1.],
                   [1., 1.]], dtype = np.float32)

targets = np.array([[0.],
                    [1.],
                    [1.],
                    [0.]], dtype = np.float32)

W1 = np.random.randn(2, 2)
B1 = np.random.randn(2, 1)
W2 = np.random.randn(1, 2)
B2 = np.random.randn(1, 1)

learning_rate = 0.1

def forward(input, target, W1, B1, W2, B2):
    X = np.transpose(input, (1, 0))
    G1 = np.dot(W1, X) + B1
    S1 = 1 / (1 + np.exp(-G1))
    G2 = np.dot(W2, S1) + B2
    S2 = 1 / (1 + np.exp(-G2))
    pred = np.transpose(S2, (1, 0))
    loss = -target * np.log(pred) - (1 - target) * np.log(1 - pred)
    loss = np.mean(loss)
    return G1, S1, G2, S2, pred, loss

def loss_gradient(input, target, W1, B1, W2, B2):
    G1, S1, G2, S2, _, _ = forward(input, target, W1, B1, W2, B2)
    target = np.transpose(target, (1, 0))
    dL_dS2 = -(target / S2) + ((1 - target) / (1 - S2))
    dS2_dG2 = S2 * (1 - S2)
    dG2_dS1 = np.transpose(W2, (1, 0))
    dG2_dW2 = np.transpose(S1, (1, 0))
    dG2_dB2 = np.ones_like(G2) 
    dS1_dG1 = S1 * (1 - S1)
    dG1_dW2 = input
    dG1_dB1 = np.ones_like(G1)

    dL_dG2 = dL_dS2 * dS2_dG2
    dL_dS1 = np.dot(dG2_dS1, dL_dG2)
    dL_dG1 = dL_dS1 * dS1_dG1
    dL_dW1 = np.dot(dL_dG1, dG1_dW2)

    dL_dB1 = np.sum(dL_dG1, axis=1, keepdims=True)
    dL_dW2 = np.dot(dL_dG2, dG2_dW2)
    dL_dB2 = np.sum(dL_dG2).reshape(1, 1)

    return dL_dW1, dL_dB1, dL_dW2, dL_dB2

_, _, _, _, pred, loss = forward(inputs, targets, W1, B1, W2, B2)
print('before pred: ', pred)
print('before loss: ', loss)

for i in range(2000):
    dL_dW1, dL_dB1, dL_dW2, dL_dB2 = loss_gradient(inputs, targets, W1, B1,W2, B2)
    W1 = W1 + -1*learning_rate*dL_dW1
    B1 = B1 + -1*learning_rate*dL_dB1
    W2 = W2 + -1*learning_rate*dL_dW2
    B2 = B2 + -1*learning_rate*dL_dB2

_, _, _, _, pred, loss = forward(inputs, targets, W1, B1, W2, B2)
print('after pred: ', pred)
print('after loss: ', loss)

