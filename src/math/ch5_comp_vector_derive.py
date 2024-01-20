
# https://toyourlight.tistory.com/8
# 백터 합성 함수의 도함수 표현

import numpy as np

X = np.array([[1, 2, 3]])
W = np.array([[3],
             [2],
             [1]])

#X = np.transpose(X, (1, 0))
#W = np.transpose(W, (1, 0))
#print('X transpose', X)
#print('W transpose', W)

def matmul_forward(X, W):
    return np.dot(X, W)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv(func, input, delta=0.001):
    return (func(input + delta) - func(input)) / delta

def matrix_sigmoid(X, W, func_list):
    f1 = func_list[0]
    f2 = func_list[1]

    F1 = f1(X, W)
    F2 = f2(F1)
    return F2


def matrix_sigmoid_deriv(X, W, func_list):
    f1 = func_list[0] #matmul
    f2 = func_list[1] #sigmoid

    F1 = f1(X, W)

    #∂(f2(f1))/∂(f1)
    df2f1_df1 = deriv(f2, f1(X, W))

    #∂(f1)/∂X
    df1_dX = np.transpose(W, (1, 0))

    #∂(f1)/∂W
    df1_dW = np.transpose(X, (1, 0))

    #∂(f2(f1))/∂x = ∂(f2(f1))/∂(f1) x ∂(f1)/∂x
    #∂(f2(f1))/∂y = ∂(f2(f1))/∂(f1) x ∂(f1)/∂y
    return df2f1_df1 * df1_dX, df2f1_df1 * df1_dW

list_func = [matmul_forward, sigmoid]

x2 = np.array([[1.001, 2., 3.]])

print('X: ', X)
print('W: ', W)
print('x2: ', x2)

F_x2 = matrix_sigmoid(x2, W, list_func)
F_x = matrix_sigmoid(X, W, list_func)
print('f(X2, W) :', F_x2)
print('f(X, W) : ', F_x)

dF_dX, dF_dW = matrix_sigmoid_deriv(X, W, list_func)
print('∂(f1(X, W))/∂W : ', dF_dX)

target = F_x2 - F_x
pred = dF_dX[0][0] *0.001
print('target(f(X2, W) - f(X, W))', target[0][0])
print('pred(∂(f(X, W))/∂x X ∂x)', pred)



