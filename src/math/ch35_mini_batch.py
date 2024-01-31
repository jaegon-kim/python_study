import numpy as np 
from numpy import genfromtxt
import matplotlib.pyplot as plt

np.random.seed(220106)

data = genfromtxt('IRIS_onehot.csv', delimiter=',', skip_header=1)
np.random.shuffle(data)

inputs = data[:130, 0:4]
targets = data[:130, 4:7]

test_inputs = data[130:, 0:4]
test_targets = data[130:, 4:7]

W = np.random.randn(3, 4)
B = np.random.randn(3, 1)

learning_rate = 0.001

batch_size = 64
steps = 0

def make_batch(input, target, step, batch_size):
    if len(input) >= step + batch_size:
        input_batch = input[step:step + batch_size]
        target_batch = target[step:step + batch_size]
    else:
        input_batch = input[step : ]
        target_batch = target[step : ]
    return input_batch, target_batch

def forward(input, target, W, B):
    X = np.transpose(input, (1, 0))
    G = np.dot(W, X) + B
    exp = np.exp(G)
    sum_exp = np.sum(exp, axis = 0, keepdims=True)
    S = exp/sum_exp
    pred = np.transpose(S, (1, 0))
    target_Y = np.sum(pred * target, axis=1, keepdims=True)
    losses = np.sum(-np.log(target_Y))
    return G, S, target_Y, pred, losses

def backward(G, S, target_Y, input, target):
    dL_dS = -1/(np.transpose(target_Y, (1, 0)))
    grads = []
    for s in np.transpose(S, (1, 0)):
        grad_matrix = np.zeros((s.size, s.size))
        for i in range(len(S)):
            for j in range(len(s)):
                if i == j:
                    grad_matrix[i][j] = s[i]*(1-s[i])
                else:
                    grad_matrix[i][j] = -s[i]*s[j]
        grads.append(grad_matrix.tolist())

    grads = np.array(grads)
    Y_target = np.expand_dims(target, axis=1)

    dS_dG = []

    for i in range(len(Y_target)):
        value = np.dot(grads[i], np.transpose(Y_target[i], (1, 0)))
        value = np.transpose(value, (1, 0))
        dS_dG.append(value.tolist())

    dS_dG = np.array(dS_dG)
    dS_dG = dS_dG.reshape(len(Y_target), -1)
    dS_dG = np.transpose(dS_dG, (1,0))

    dG_dW = input
    dG_dB = np.ones_like(G)
    dL_dG = dL_dS * dS_dG
    dL_dW = np.dot(dL_dG, dG_dW)
    dL_dB = np.sum(dL_dG * dG_dB, axis=1, keepdims=True)

    return dL_dW, dL_dB

_, _, _, pred, losses = forward(inputs, targets, W, B)
print('before loss:', losses)

arr_loss = []
for i in range(500):
    if steps > len(inputs):
        steps = 0
    x_batch, y_batch = make_batch(inputs, targets, steps, batch_size)
    G1, S, target_Y, pred, loss = forward(x_batch, y_batch, W, B)
    dL_dW, dL_dB = backward(G1, S, target_Y, x_batch, y_batch)
    W = W + -1 * learning_rate * dL_dW
    B = B + -1 * learning_rate * dL_dB
    steps += batch_size
    arr_loss.append(loss)

_, _, _, pred, losses = forward(inputs, targets, W, B)
print('before loss:', losses)    

_, _, _, test_pred, losses = forward(test_inputs, test_targets, W, B)
equal_num = np.argmax(test_pred, axis=1) == np.argmax(test_targets, axis=1)
print('equal num:', equal_num)
print('accuracy', np.sum(equal_num/len(equal_num)))

filter_test_pred = np.where(test_pred >= 0.7, 1, 0)
filter_test_targets = np.array(test_targets, dtype=np.int16)
filter_equal_num = np.argmax(filter_test_pred, axis=1) == np.argmax(filter_test_targets, axis=1)
print('filter equal_num', filter_equal_num)
print('filter accurac', np.sum(filter_equal_num/len(filter_equal_num)))

plt.plot(arr_loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

