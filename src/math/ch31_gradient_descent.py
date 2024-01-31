import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

fig = plt.figure()
ax = plt.axes(xlim=(-10, 10))
start_x = -4

learning_rate = 1.05

def loss_function(x):
    return x**2

def loss_gradient(x):
    return 2*x

xrange = np.arange(-10, 10, 0.2)
loss_array = []

for x in xrange:
    loss = loss_function(x)
    loss_array.append(loss)

ax.plot(xrange, loss_array)
ax.set_xlabel('x')
ax.set_ylabel('loss')

train_x = []
for i in range(20):
    dL_dx = loss_gradient(start_x)
    start_x = start_x + -1 * learning_rate * dL_dx
    train_x.append(start_x)

print('train_x: ', train_x)

redDot, = ax.plot([], [], 'ro')

def animate(frame):
    loss = loss_function(frame)
    redDot.set_data(frame, loss)
    return redDot

ani = FuncAnimation(fig, animate, frames=train_x)


plt.show()