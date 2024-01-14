import numpy as np 


// https://toyourlight.tistory.com/4

def linear(x):
    return 2 * x

def square(x):
    return np.power(x, 2)

def comp_func(list_func, x):
    f1 = list_func[0]
    f2 = list_func[1]
    return f2(f1(x))

print("Hello World")
print(linear(10))
print(square(10))

func_list = [linear, square]
print(comp_func(func_list, 2))