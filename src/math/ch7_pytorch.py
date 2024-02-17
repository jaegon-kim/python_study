import numpy as np
import torch

X = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6],
              [0.7, 0.8, 0.9]])
W = np.array([[1., 2.],
              [3., 4.],
              [5., 6.]])

# requires_grad는 pyTorch의 Autograd(자동미분)을 활성화하는 데 사용된다.
X = torch.tensor(X, requires_grad = True)
W = torch.tensor(W, requires_grad = False)

#g(X, W) = X x W
g_XW = torch.matmul(X, W)

#σ(x) = 1 / (1 + exp^(-x))  : sigmoid 
s_XW = 1 / (1 + torch.exp(-g_XW))

#h(x) = Σx
h_XW = torch.sum(s_XW)

# h (σ( g(X, W) ) ) 를 X로 미분
h_XW.backward()

print(X.grad)