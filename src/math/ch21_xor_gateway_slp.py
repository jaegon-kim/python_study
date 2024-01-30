import numpy as np 

np.random.seed(220111)

input = np.array([[0., 0.],
                  [1., 0.],
                  [0., 1.],
                  [1., 1.]], dtype = np.float32)

target_or = np.array([[0.],
                      [1.],
                      [1.],
                      [1.]], dtype = np.float32)

target_and = np.array([[0.],
                       [0.],
                       [0.],
                       [1.]], dtype = np.float32)

target_nand = np.array([[1.],
                        [1.],
                        [1.],
                        [0.]], dtype = np.float32)

target_xor = np.array([[0.],
                       [1.],
                       [1.],
                       [0.]], dtype = np.float32)

def logistics_forward(X, Y, W, B):
    XWB = np.dot(X, W) + B
    pred = 1 / (1 + np.exp(-XWB))
    loss = np.sum(-Y * np.log(pred) - (1-Y)*np.log(1-pred))
    return pred, loss

def loss_gradient(X, Y, W, B):
    XWB = np.dot(X, W) + B
    pred = 1 / (1 + np.exp(-XWB))

    dL_dsig = -1 *( (Y/pred) - ((1 - Y) / (1 - pred) ))
    dsig_dg = ( 1 / (1 + np.exp(-XWB)) ) * (1 - 1/(1 + np.exp(-XWB)) )
    dg_dW = np.transpose(X, (1, 0))

    dloss_dW = np.dot(dg_dW, dL_dsig * dsig_dg)
    dloss_dB = np.sum(dL_dsig * dsig_dg, axis = 0)

    return dloss_dW, dloss_dB


print('------------- Learning OR ------------------ ')

W = np.random.randn(2, 1)
B = np.random.randn(1, 1)

print('W: ', W)
print('B: ', B)

learning_rate = 1


pred, loss = logistics_forward(input, target_or, W, B)
print('before pred: ', pred)
print('before loss: ', loss)

for i in range(100):
    dL_dW, dL_dB = loss_gradient(input, target_or, W, B)
    W = W + -1 * learning_rate * dL_dW
    B = B + -1 * learning_rate * dL_dB

pred, loss = logistics_forward(input, target_or, W, B)
print('after pred: ', pred)
print('after loss: ', loss)

print('W: ', W)
print('B: ', B)


print('------------- Learning NAND ------------------ ')

W = np.random.randn(2, 1)
B = np.random.randn(1, 1)

print('W: ', W)
print('B: ', B)

learning_rate = 1


pred, loss = logistics_forward(input, target_nand, W, B)
print('before pred: ', pred)
print('before loss: ', loss)

for i in range(100):
    dL_dW, dL_dB = loss_gradient(input, target_nand, W, B)
    W = W + -1 * learning_rate * dL_dW
    B = B + -1 * learning_rate * dL_dB

pred, loss = logistics_forward(input, target_nand, W, B)
print('after pred: ', pred)
print('after loss: ', loss)

print('W: ', W)
print('B: ', B)


print('------------- Learning XOR ------------------ ')

W = np.random.randn(2, 1)
B = np.random.randn(1, 1)

print('W: ', W)
print('B: ', B)

pred, loss = logistics_forward(input, target_xor, W, B)
print('before pred: ', pred)
print('before loss: ', loss)

for i in range(100):
    dL_dW, dL_dB = loss_gradient(input, target_xor, W, B)
    W = W + -1 * learning_rate * dL_dW
    B = B + -1 * learning_rate * dL_dB

pred, loss = logistics_forward(input, target_xor, W, B)
print('after pred: ', pred)
print('after loss: ', loss)


print('W: ', W)
print('B: ', B)




