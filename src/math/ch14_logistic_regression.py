#https://toyourlight.tistory.com/14

import numpy as np 
from numpy import genfromtxt

np.random.seed(220104)

data = genfromtxt('NBA_Rookie_draft.csv', delimiter=',', skip_header = 1)
input = data[:, 0:3] # 모든 행에 대해서, 컬럼 0부터 3 앞까지 (0, 1, 2)
target = data[:, -1:] # 모든 행에 대해서 뒤에서 첫번째 요소

print('input : ', input)
print('target: ', target)

W = np.random.randn(3, 1)
B = np.random.randn(1, 1)

print('W: ', W)
print('B: ', B)

learning_rate = 0.001

def logistic_forward(X, Y, W, B):
    XWB = np.dot(X, W) + B
    pred = 1 / (1 + np.exp(-XWB)) # sigmoid
    loss = np.sum(-Y * np.log(pred) - (1 - Y)* np.log(1 - pred))
    return pred, loss

def loss_gradient(X, Y, W, B):

    # g(W, B)
    XWB = np.dot(X, W) + B

    # σ(g(W, B))
    pred = 1 / (1 + np.exp(-XWB))

    # ∂ L(σ(g(W, B))) / ∂σ(g(W, B))
    dL_dsig = -1 * ((Y/pred) - ((1-Y)/(1-pred)))

    # ∂ σ(g(W, B)) / ∂g(W, B)
    dsig_dg = (1/(1+np.exp(-XWB))) * (1-1/(1+np.exp(-XWB)))

    # ∂ g(W, B) / ∂W
    dg_dW = np.transpose(X, (1, 0))

    # ∂ g(W, B) / ∂B = 1

    # ∂ L(σ(g(W, B))) / ∂ W
    dloss_dW = np.dot(dg_dW, dL_dsig*dsig_dg)

    dloss_dB = np.sum(dL_dsig*dsig_dg, axis=0)

    return  dloss_dW, dloss_dB
 

pred, loss = logistic_forward(input, target, W, B)

print('before pred: ', pred)
print('bofore loss: ', loss)
print('before W: ', W)
print('before B: ', B)

for i in range(3000):
    dL_dW, dL_dB = loss_gradient(input, target, W, B)
    W = W + -1 * learning_rate * dL_dW
    B = B + -1 * learning_rate * dL_dB

pred, loss = logistic_forward(input, target, W, B)

print('after pred: ', pred)
print('after loss: ', loss)
print('after W: ', W)
print('after B: ', B)
