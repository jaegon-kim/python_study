import numpy as np 
from numpy import genfromtxt

np.random.seed(220106)

data = genfromtxt('IRIS_onehot.csv', delimiter=',', skip_header=1)
np.random.shuffle(data)

input = data[:130, 0:4] * 0.1
target = data[:130, 4:7]

test_input = data[130:, 0:4] *0.1
test_target = data[130:, 4:7]

W = np.random.randn(3, 4)
B = np.random.randn(3, 1)

learning_rate = 0.0005

batch_size = 16

def softmax_forward(X, Y, W, B):
    
    arr_pred = []
    arr_loss = []

    for i in range(len(X)):
        
        x = [X[i]]
        y = [Y[i]]

        x = np.transpose(x )
        WXB = np.dot(W, x) + B

        e_WXB1 = np.exp(WXB[0])
        e_WXB2 = np.exp(WXB[1])
        e_WXB3 = np.exp(WXB[2])

        sum_e_WXB = e_WXB1 + e_WXB2 + e_WXB3

        smax_WXB1 = e_WXB1 / sum_e_WXB
        smax_WXB2 = e_WXB2 / sum_e_WXB
        smax_WXB3 = e_WXB3 / sum_e_WXB

        softmax = np.array([smax_WXB1[0], smax_WXB2[0], smax_WXB3[0]])

        arr_pred.append(softmax)

        y_target = np.sum(softmax * y, axis = 1, keepdims=True)

        loss = -np.log(y_target)
        arr_loss.append(loss)
    
    preds = np.array(arr_pred)
    losses = np.sum(np.array(arr_loss))

    return preds, losses

def loss_gradient(X, Y, W, B):

    add_dL_dW = np.zeros((3, 4))
    add_dL_dB = np.zeros((3, 1))

    for i in range(len(X)):

        x = [X[i]]
        y = [Y[i]]

        x = np.transpose(x )
        WXB = np.dot(W, x) + B

        e_WXB1 = np.exp(WXB[0])
        e_WXB2 = np.exp(WXB[1])
        e_WXB3 = np.exp(WXB[2])

        sum_e_WXB = e_WXB1 + e_WXB2 + e_WXB3

        smax_WXB1 = e_WXB1 / sum_e_WXB
        smax_WXB2 = e_WXB2 / sum_e_WXB
        smax_WXB3 = e_WXB3 / sum_e_WXB

        softmax = np.array([[smax_WXB1[0], smax_WXB2[0], smax_WXB3[0]]])

        y_target = np.sum(softmax * y, axis=1, keepdims=True)

        # ∂L(smax(g(W, B))) / ∂smax(g(W,B))
        dL_dsmax = -1 / y_target

        # derivation of smax matrix (3x3)
        dsmax_dg_matrix = np.array(
            [
                [(smax_WXB1 * (1 - smax_WXB1))[0], -(smax_WXB1 * smax_WXB2)[0], -(smax_WXB1 * smax_WXB3)[0]],
                [-(smax_WXB1 * smax_WXB2)[0], (smax_WXB2 * (1 - smax_WXB1))[0], -(smax_WXB2 * smax_WXB3)[0]],
                [-(smax_WXB1 * smax_WXB3)[0], -(smax_WXB2 * smax_WXB3)[0], (smax_WXB2 * (1 - smax_WXB1))[0]]
            ]
        )

        # ∂ smax(g(W, B)) / ∂g(W, B)
        dsmax_dg = np.dot(dsmax_dg_matrix, np.transpose(y, (1, 0)))

        # ∂g(W, B) / ∂W
        dg_dW = np.transpose(x, (1, 0))

        # ∂smax(g(W,B)) / ∂W = ∂smax(g(W,B))/∂g(W,B) * ∂g(W,∂) / ∂W
        dsmax_dW = np.dot(dsmax_dg, dg_dW)

        # ∂L(g(W,B))/∂W = ∂L/∂smax(g(W,B)) * ∂smax(g(W,B))/∂W
        dL_dW = dL_dsmax * dsmax_dW

        # 
        add_dL_dW = add_dL_dW + dL_dW

        # ∂L(g(W,B))/∂B = ∂L/∂smax(g(W,B)) * ∂smax(g(W,B))/∂g(W,B)*∂g(W,B)/∂B
        dL_dB = dL_dsmax * dsmax_dg

        #
        add_dL_dB = add_dL_dB + dL_dB

    return add_dL_dW, add_dL_dB


pred, loss = softmax_forward(input, target, W, B)
#print('before pred: ', pred)
print('before loss: ', loss)
#print('before W: ', W)
#print('before B: ', B)

for i in range(2000):
    dL_dW, dL_dB = loss_gradient(input, target, W, B)
    W = W + -1 * learning_rate * dL_dW
    B = B + -1 * learning_rate * dL_dB

pred, loss = softmax_forward(input, target, W, B)
#print('after pred: ', pred)
print('after loss: ', loss)
#print('after W: ', W)
#print('after B: ', B)

test_pred, _ = softmax_forward(test_input, test_target, W, B)
#print('test_pred: ', test_pred)
#print('np.argmax(test_pred, axis=1):', np.argmax(test_pred, axis=1))
#print('test_target: ', test_target)
#print('np.argmax(test_pred, axis=1):', np.argmax(test_target, axis=1))


equal_num = np.argmax(test_pred, axis=1) == np.argmax(test_target, axis=1)
print('equal_num: ', equal_num)
print('accuracy: ', np.sum(equal_num/len(equal_num)))

filter_test_pred = np.where(test_pred >= 0.7, 1., 0.)
filter_equal_num = np.argmax(filter_test_pred, axis=1) == np.argmax(test_target, axis=1)
print('filter equal_num: ', filter_equal_num)
print('filter accuracy:', np.sum(filter_equal_num/len(filter_equal_num)))



