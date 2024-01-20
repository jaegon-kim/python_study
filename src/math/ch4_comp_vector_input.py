import numpy as np

X = np.array([[1, 2, 3]])
W = np.array([[3],
              [2],
              [1]])
print('x.shape', X.shape)
print('w.shape', W.shape)

def matmul_forward(X, W):
    return np.dot(X, W)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def matrix_sigmoid(X, W, func_list):
    f1 = func_list[0]
    f2 = func_list[1]

    F1 = f1(X, W)
    F2 = f2(F1)
    return F2

list_func = [matmul_forward, sigmoid]

print('matmtul_forward ', matmul_forward(X, W))

print('matrix_sigmoid', matrix_sigmoid(X, W, list_func))