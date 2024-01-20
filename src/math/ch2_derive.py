#import math
import numpy as np
import matplotlib.pyplot as plt

# https://toyourlight.tistory.com/5

# 미분(differentiation), 도함수(derivative)
# 함수 y=f(x)에서 도함수 f'(x)를 구하는 것을 함수 f(x)를 x에 대하여 미분한다고하며 그 계산법을 미분법이라고 한다.

def log(x):
    return np.log(x)

def square(x):
    return np.power(x, 2)

def derive(func, input, delta=0.001):
    return (func(input + delta) - func(input)) / delta


#print(log(10))

#x = np.arange(-2, 2, 0.01)
x = np.arange(0.01, 2, 0.01)
plt.plot(x, square(x), 'r', label='square')
plt.plot(x, log(x), 'g', label='log')
plt.plot(x, derive(square, x), 'b', label='drive_square')
plt.plot(x, derive(log, x), 'y', label='drive_log')
plt.legend()
plt.grid(True)
plt.show()
