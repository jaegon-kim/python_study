import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from numpy import genfromtxt
import matplotlib.pyplot as plt 

torch.manual_seed(220119)

data = genfromtxt('weight_height.csv', delimiter=',', skip_header = 1)
data_input = data[:, 0].reshape(-1, 1)
data_target = data[:, 1].reshape(-1, 1)

input = torch.FloatTensor(data_input) * 0.1
target = torch.FloatTensor(data_target) * 0.1

model = nn.Sequential(
    nn.Linear(1, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

pred = model(input)
loss = F.mse_loss(pred, target)
print('before loss: ', loss)
optimizer = optim.SGD(model.parameters(), lr = 0.01)
epoches = 3000

for epoch in range(epoches + 1):
    pred = model(input)
    loss = F.mse_loss(pred, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


pred = model(input)
loss = F.mse_loss(pred, target)
print('after loss: ', loss)
print(list(model.parameters()))

plt.plot(input*10, target*10, 'ro', label='target')
plt.plot(input*10, pred.detach()*10, 'bo', label='pred')
plt.xlabel('weight', size=15)
plt.ylabel('height', size=15)
plt.legend()
plt.show()

