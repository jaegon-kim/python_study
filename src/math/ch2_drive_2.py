import numpy as np
import matplotlib.pyplot as plt

def linear(x):
    return 2 * x

def square(x):
    return np.power(x, 2)

def derive(func, input, delta=0.001):
    return (func(input + delta) - func(input)) / delta

def comp_func(list_func, x):
    f1 = list_func[0]
    f2 = list_func[1]
    return f2(f1(x))

def chain_derive(chain, input):
    f1 = chain[0]
    f2 = chain[1]
    
    f1_x = f1(input)

    df1_dx = derive(f1, input)

    df2_df1 = derive(f2, f1_x)

    return df2_df1 * df1_dx

func_list = [linear, square]

x = np.arange(-2, 2, 0.01)
plt.plot(x, comp_func(func_list, x), 'r', label='comp_func')
plt.plot(x, chain_derive(func_list, x), 'b', label='chain_derive')
plt.legend()
plt.grid(True)
plt.show()
