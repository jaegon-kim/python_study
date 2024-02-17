import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(220102)

input = np.array([[1.],
                  [2.],
                  [3.]])

target = np.array([[3.],
                   [5.],
                   [7.]])

W = np.random.randn(1, 1) # Correct value = 2
B = np.random.randn(1, 1) # Correct value = 1

learning_rate = 0.001

# 예상 값, W, B를 입력으로 받아 예측값(pred)와 오차(loss)를 반환한다. 
def linear_forward(X, Y, W, B): #Y: Original 값
    XW = np.dot(X, W)
    pred = XW + B # 예측값
    loss = np.mean(np.power(Y - pred, 2)) # (예측 값과 Original 값과의 차의 제곱에 대한 평균)
    return pred, loss

# 오차(loss) 함수에 대한 도함수(W에 대한 편미분, B에 대한 편미분)들을 반환한다.
def loss_gradient(X, Y, W, B):
    XWB = np.dot(X, W) + B # X x W + B

    # ∂L(g(X, W, B)) / ∂g(X, W, B)
    dL_dg = 2 * (XWB - Y) # ..? sum은 어디 간 것일까 ?

    # ∂g(X, W, B) / ∂W 
    dg_dW = np.transpose(X, (1, 0)) # 전치행렬 ?

    # ∂L(g(X, W, B)) / ∂W
    dL_dW = np.dot(dg_dW, dL_dg)

    # ∂L(g(X, W, B)) / dB
    dL_dB = np.sum(dL_dg, axis = 0)

    return dL_dW, dL_dB

pred, loss = linear_forward(input, target, W, B)

print('random W: ', W)
print('random B: ', B)
print('pred    : ', pred)
print('loss    : ', loss)


dL_dW, dL_dB = loss_gradient(input, target, W, B )
print('before weight :', W)
print('before bias   :', B)

train_W = []
for i in range(100): 
    dL_dW, dL_dB = loss_gradient(input, target, W, B )
    W = W + -1 * learning_rate * dL_dW
    B = B + -1 * learning_rate * dL_dB
    train_W.append(W)

print('after weight :', W)
print('after bias   :', B)

pred, loss = linear_forward(input, target, W, B)
print('pred    : ', pred)
print('loss    : ', loss)


# Checking linear regression with graph !
# The W near '2' has almost minimum loss

weights = np.arange(0, 5, 0.2)
loss_array = []

for weight in weights:
    _, loss = linear_forward(input, target, weight, B)
    loss_array.append(loss)

fig = plt.figure()
ax = plt.axes(xlim=(-0, 4))
ax.plot(weights, loss_array)
ax.set_xlabel('weight')
ax.set_ylabel('loss')
#plt.show()

redDot, = plt.plot([], [], 'ro')

def animate(frame):
    _, loss = linear_forward(input, target, frame, B)
    redDot.set_data(frame, loss)
    return redDot

ani = FuncAnimation(fig, animate, frames=train_W)

plt.title("Mean Square Error Loss Function")
plt.show()