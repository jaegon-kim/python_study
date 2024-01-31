import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from numpy import genfromtxt
import numpy as np 
import matplotlib.pyplot as plt 


torch.manual_seed(220214)
data = genfromtxt('concrete_data.csv', delimiter=',', skip_header=1)
#print('data: ', data)

norm_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

train_data = torch.FloatTensor(norm_data[:1000, :])
test_data = torch.FloatTensor(norm_data[1000:, :])

inputs = train_data[:, 0:8]
targets = train_data[:, -1:]

test_inputs = test_data[:, 0:8]
test_targets = test_data[:, -1:]

'''
model = nn.Sequential(
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)
'''
model = nn.Sequential(
    nn.Linear(8, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)

pred = model(inputs)
loss = F.mse_loss(pred, targets)
print('before loss: ', loss)

#optimizer = optim.SGD(model.parameters(), lr=0.15)
optimizer = optim.SGD(model.parameters(), lr=0.2)

epoches = 3000

for epoche in range(epoches + 1):
    pred = model(inputs)
    loss = F.mse_loss(pred, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

pred = model(inputs)
loss = F.mse_loss(pred, targets)
print('after loss: ', loss)

test_pred = model(test_inputs)

test_targets = test_targets * np.std(data[:, -1:], axis=0) + np.mean(data[:, -1:], axis=0)
test_pred = test_pred.detach() * np.std(data[:, -1:], axis=0) + np.mean(data[:, -1:], axis=0)
plt.plot(test_targets, 'ro', label='target')
plt.plot(test_pred, 'bo', label='pred')
plt.legend()
plt.show()

