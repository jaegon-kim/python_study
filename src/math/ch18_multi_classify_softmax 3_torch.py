import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from numpy import genfromtxt

data = genfromtxt('IRIS_tiny_onehot.csv', delimiter=',', skip_header=1)

input = data[:, 0:4]
target = data[:, -1:]

input = torch.FloatTensor(input)
target = torch.LongTensor(target).squeeze()

model = nn.Sequential(
    nn.Linear(4, 3),
    nn.Softmax(dim=1)
)

pred = model(input)
print('before pred: ', pred)
loss = F.cross_entropy(pred, target)

print('before loss: ', loss)
optimizer = optim.SGD(model.parameters(), lr=0.1)
epoches = 3000

for epoche in range(epoches + 1):
    pred = model(input)
    loss = F.cross_entropy(pred, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

pred = model(input)
print('after pred: ', pred)
loss = F.cross_entropy(pred, target)
print('after loss: ', loss)
print(list(model.parameters()))