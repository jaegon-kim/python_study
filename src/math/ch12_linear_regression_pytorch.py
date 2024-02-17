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

print('before pred: \n', pred)

for i in range(1000):
    pred = torch.matmul(input, W) + B
    loss = torch.mean(torch.pow((pred - target), 2))

    if i % 100 == 0:
        print('loss: ', loss.shape,  loss)

    loss.backward()

    # 사실 Pytorch에서 내부적으로 W, B를 관리하지만 동작을 이해하기 위해 외부에 꺼내 놓고 한다.
    W.data = W.data - learning_rate * W.grad.data
    B.data = B.data - learning_rate * B.grad.data

    # 파라미터 W, B를 밖으로 꺼내서 쓰고 있으니까, 누적을 할 필요가 없다. 기울기 값을 초기화 시킨다. 
    W.grad.zero_()
    B.grad.zero_() 

pred = torch.matmul(input, W) + B

print('after pred: \n', pred)

