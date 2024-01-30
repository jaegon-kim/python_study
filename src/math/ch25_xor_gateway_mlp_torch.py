import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim


input = torch.FloatTensor([[0., 0.],
                   [1., 0.],
                   [0., 1.],
                   [1., 1.]])

target = torch.FloatTensor([[0.],
                    [1.],
                    [1.],
                    [0.]])

model = nn.Sequential(
    nn.Linear(2, 2),
    nn.Sigmoid(),
    nn.Linear(2, 1),
    nn.Sigmoid()
)

pred = model(input)
print('before pred: ', pred)
optimizer = optim.SGD(model.parameters(), lr = 1) #SGD Stochastic Gradient Descent 확율적 경사 하강법
epoches = 2000

for epoche in range(epoches + 1):
    pred = model(input)
    loss = F.binary_cross_entropy(pred, target)

    optimizer.zero_grad() # SGD에서 gradient를 0으로 초기화 하는 역할
    loss.backward()
    optimizer.step()
    if epoche % 100 == 0:
        print('loss: ', loss)

pred = model(input)
print('after pred: ', pred)
