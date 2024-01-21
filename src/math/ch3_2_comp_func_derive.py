# https://toyourlight.tistory.com/6
# Deriving composite functions 

from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
fontlabel = {"fontsize":"large", "color":"gray", "fontweight":"bold"}
ax.set_xlabel("X", fontdict=fontlabel, labelpad=16)
ax.set_ylabel("Y", fontdict=fontlabel, labelpad=16)


def multiple_func(x, y):
#    return x + y
    return x * y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def comp_func(list_func, x, y):
    f1 = list_func[0]
    f2 = list_func[1]
    return f2(f1(x, y))

def derive(func, input, delta=0.01):
    return (func(input + delta) - func(input)) / delta

def chain_multiple_deriv_add(list_func, x, y):
    f1 = list_func[0] # x + y
    f2 = list_func[1] # sigmoid:1 / (1 + np.exp(-x))

    # ∂(f1)/∂x : f1 = x + y를 x로 편미분
    df1_dx = 1
    #df1_dx = y

    # ∂(f1)/∂y : f1 = x + y를 y로 편미분
    df1_dy = 1
    #df1_dy = x

    # ∂(f2(f1))/∂x(f1)
    df2f1_df1 = derive(f2, f1(x, y))

    # ∂(f2(f1))/∂x = ∂(f2(f1))/∂(f1) x ∂(f1)/∂x : 합성 함수에 chain rule 적용
    # ∂(f2(f1))/∂y = ∂(f2(f1))/∂(f1) x ∂(f1)/∂y : 합성 함수에 chain rule 적용
    return df2f1_df1 * df1_dx, df2f1_df1 * df1_dy

func_list = [multiple_func, sigmoid]

X = np.arange(-3, 3, 0.01)
Y = np.arange(-3, 3, 0.01)
Z = comp_func(func_list, X, Y)
ax.plot(X, Y, Z, 'r', label='comp_func')

Z_dx, Z_dy = chain_multiple_deriv_add(func_list, X, Y)
ax.plot(X, Y, Z_dx, 'b', label = 'Z_dx')
ax.plot(X, Y, Z_dy, 'g', label = 'Z_dy')

plt.tight_layout()
plt.legend()
plt.show()



