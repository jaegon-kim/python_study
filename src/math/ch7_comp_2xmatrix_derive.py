import numpy as np

X = np.array([[0.1, 0.2, -.3],
              [0.4, 0.5, 0.6],
              [0.7, 0.8, 0.9]])

W = np.array([[1, 2],
              [3, 4],
              [5, 6]])

X2 = np.array([[0.1001, 0.2, 0.3],
               [0.4, 0.5, 0.6],
               [0.7, 0.8, 0.9]])

def f_forward(X, W):

    # g(X, W) = X x W
    g_XW = np.dot(X, W)

    # σ(X) = 1 / (1 + exp^(-x))
    s_XW = 1 / (1 + np.exp(-g_XW))

    h_XW = np.sum(s_XW)

    return h_XW

def deriv(func, input, delta=0.001):
    return (func(input + delta) - func(input)) / delta

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def f_backward(X, W, sig):
    #g(X, W) = X x W, (3, 2) shape
    g_XW = np.dot(X, W)

    #dσ(g(X, W))/dg(X, W)
    dsdg = deriv(sig, g_XW)

    #dg(X, W)/dX
    dgdx = np.transpose(W, (1, 0))

    #dh(σ(g(X, W)))/dx
    dfdx = np.dot(dsdg, dgdx)

    return dfdx

print('X: ', X)
print('f(X) :', f_forward(X, W))
print('df(x)/dX:', f_backward(X, W, sigmoid))
print(' ')
print('X2: ', X2)
print('f(X2): ', f_forward(X2, W))
