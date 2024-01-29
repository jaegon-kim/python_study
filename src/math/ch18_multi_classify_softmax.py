import numpy as np 

np.random.seed(220106)

input = np.array([[1.2, 2.4, 3.6, 1.8]], dtype=np.float32)
target = np.array([[0, 0, 1]], dtype=np.int32)

W = np.random.randn(3, 4)
B = np.random.randn(3, 1)

learning_rate = 0.001

def softmax_forward(X, Y, W, B):
    X = np.transpose(X, (1, 0)) # transpose (1,4) -> (4, 1)
    WXB = np.dot(W, X) + B

    e_WXB1 = np.exp(WXB[0])
    e_WXB2 = np.exp(WXB[1])
    e_WXB3 = np.exp(WXB[2])

    sum_e_WXB = e_WXB1 + e_WXB2 + e_WXB3

    smax_WXB1 = e_WXB1 / sum_e_WXB
    smax_WXB2 = e_WXB2 / sum_e_WXB
    smax_WXB3 = e_WXB3 / sum_e_WXB

    pred = np.array([[smax_WXB1[0], smax_WXB2[0], smax_WXB3[0]]])

    Y_target = np.sum(pred * Y, axis = 1, keepdims=True)

    loss = -np.log(Y_target)

    return pred, loss

def loss_gradient(X, Y, W, B):
    X = np.transpose(X, (1, 0))
    WXB = np.dot(W, X) + B

    e_WXB1 = np.exp(WXB[0])
    e_WXB2 = np.exp(WXB[1])
    e_WXB3 = np.exp(WXB[2])

    sum_e_WXB = e_WXB1 + e_WXB2 + e_WXB3

    smax_WXB1 = e_WXB1 / sum_e_WXB
    smax_WXB2 = e_WXB2 / sum_e_WXB
    smax_WXB3 = e_WXB3 / sum_e_WXB

    softmax = np.array([[smax_WXB1[0], smax_WXB2[0], smax_WXB3[0]]])

    Y_target = np.sum(softmax * Y, axis=1, keepdims=True)

    # ∂L(smax(g(W, B))) / ∂smax(g(W,B))
    dL_dsmax = -1 / Y_target[0][0]

    # derivation of smax matrix (3x3)
    dsmax_dg_matrix = np.array(
        [
            [(smax_WXB1 * (1 - smax_WXB1))[0], -(smax_WXB1 * smax_WXB2)[0], -(smax_WXB1 * smax_WXB3)[0]],
            [-(smax_WXB1 * smax_WXB2)[0], (smax_WXB2 * (1 - smax_WXB1))[0], -(smax_WXB2 * smax_WXB3)[0]],
            [-(smax_WXB1 * smax_WXB3)[0], -(smax_WXB2 * smax_WXB3)[0], (smax_WXB2 * (1 - smax_WXB1))[0]]
        ]
    )

    # ∂ smax(g(W, B)) / ∂g(W, B)
    dsmax_dg = np.dot(dsmax_dg_matrix, np.transpose(Y, (1, 0)))

    # ∂g(W, B) / ∂W
    dg_dW = np.transpose(X, (1, 0))

    # ∂ loss(W, B) / ∂W
    dloss_dW = dL_dsmax * np.dot(dsmax_dg, dg_dW)

    # ∂ loss(W, B) / ∂B
    dloss_dB = dL_dsmax * dsmax_dg

    return dloss_dW, dloss_dB

pred, loss = softmax_forward(input, target, W, B)
print('before pred: ', pred)
print('before loss: ', loss)
print('before W: ', W)
print('before B: ', B)

for i in range(100):
    dL_dW, dL_dB = loss_gradient(input, target, W, B)
    W = W + -1 * learning_rate * dL_dW
    B = B + -1 * learning_rate * dL_dB

pred, loss = softmax_forward(input, target, W, B)
print('after pred: ', pred)
print('after loss: ', loss)
print('after W: ', W)
print('after B: ', B)