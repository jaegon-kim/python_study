import torch
import numpy as np

np.random.seed(220102)

input = np.array([[1.],
                  [2.],
                  [3.]])

target = np.array([[3.],
                   [5.],
                   [7.]])

W = np.random.randn(1, 1) # Correct value = 2
B = np.random.randn(1, 1) # Correct value = 1

input = torch.tensor(input, requires_grad = False)
target = torch.tensor(target, requires_grad = False)
W = torch.tensor(W, requires_grad=True)
B = torch.tensor(B, requires_grad=True)

learning_rate = 0.01

pred = torch.matmul(input, W) + B

print('before pred: ', pred)

for i in range(1000):
    pred = torch.matmul(input, W) + B
    loss = torch.mean(torch.pow((pred - target), 2))

    loss.backward()

    W.data = W.data - learning_rate * W.grad.data
    B.data = B.data - learning_rate * B.grad.data

    W.grad.zero_()
    W.grad.zero_()

pred = torch.matmul(input, W) + B

print('after pred: ', pred)

