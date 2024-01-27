# https://toyourlight.tistory.com/6
# composite function
# func 1: g(x, y) = x + y
# func 2: f(x) = sigmoid(@) = 1 / (1 + exp(-x))
# composite: g(x) -> f(x)  f(g(x, y)) = 1 / (1 + exp(x+y))

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

fontlabel = {"fontsize":"large", "color":"gray", "fontweight":"bold"}
ax.set_xlabel("X", fontdict=fontlabel, labelpad=16)
ax.set_ylabel("Y", fontdict=fontlabel, labelpad=16)


def multiple_func(x, y):
    return x + y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def comp_func(list_func, x, y):
    f1 = list_func[0]
    f2 = list_func[1]
    return f2(f1(x, y))

func_list = [multiple_func, sigmoid]


X = np.arange(-3, 3, 0.01)
Y = np.arange(-3, 3, 0.01)
Z = comp_func(func_list, X, Y)
ax.plot(X, Y, Z, 'r', label = 'comp_func')

plt.tight_layout()
plt.legend()
plt.show()

