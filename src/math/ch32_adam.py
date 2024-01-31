import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation 

fig = plt.figure()
ax = plt.axes(xlim=(-10, 10))
start_x = -8
learning_rate=0.3

def loss_func(x):
    return (1/200)*(x-9)*(x)*(x+1)*(x+6) + 4

def loss_gradient(x):
    return (1/100)*(2*np.power(x, 3) - 3*np.power(x, 2) - 57 * x - 27)

xrange = np.arange(-10, 10, 0.1)
loss_array = []

for x in xrange:
    loss = loss_func(x)
    loss_array.append(loss)

ax.plot(xrange, loss_array)
ax.set_xlabel('x')
ax.set_ylabel('loss')

train_x = []
beta1 = 0.9
beta2 = 0.999
v = 0
s = 0

for i in range(100):
    if i == 0:
        pass
    else:
        dL_dx = loss_gradient(start_x)
        v = beta1 * v + (1 - beta1) * dL_dx
        s = beta2 * s + (1 - beta2) * dL_dx**2
        v_avg = v/(1 - np.power(beta1, i))
        s_avg = s/(1 - np.power(beta2, i))
        start_x = start_x - learning_rate * (1/(np.sqrt(s_avg) + 1e-10)) * v_avg
    train_x.append(start_x)

redDot, = ax.plot([], [], 'ro')

def animate(frame):
    loss = loss_func(frame)
    redDot.set_data(frame, loss)
    return redDot

ani = FuncAnimation(fig, animate, frames=train_x)
plt.show()
