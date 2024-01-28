import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from numpy import genfromtxt 

data = genfromtxt('NBA_Rookie_draft.csv', delimiter=',', skip_header = 1)
input = data[:, 0:3]
target = data[:, -1:]

input = torch.FloatTensor(input)
target = torch.FloatTensor(target)

model = nn.Sequential(
    nn.Linear(3, 1),
    nn.Sigmoid()
)

print(model(input))

optimizer = optim.SGD(model.parameters(), lr=0.001)
epoches = 3000

for epoche in range(epoches + 1):
    pred = model(input)
    loss = F.binary_cross_entropy(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model(input))
print(list(model.parameters()))