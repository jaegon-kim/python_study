import numpy as np
from itertools import *

np.random.seed(230907)

#dataset = ["as", "soon", "as"]
#datalist = list(set(permutations(dataset, 3)))
datalist =[["soon", "as", "as"],
           ["as", "as", "soon"],           
           ["as", "soon", "as"]]

print('datalist: ', datalist)

def serial_prep(x):
    total_prep = np.array([])
    for sentence in x:
        prep  = np.array([])
        for word in sentence:
            if word == 'as':
                word = np.array([1, 0])
            elif word == 'soon':
                word = np.array([0, 1])
            prep = np.concatenate([prep, word])
        if total_prep.size == 0:
            total_prep = np.concatenate([total_prep, prep])
        else:
            total_prep = np.vstack([total_prep, prep])
    return total_prep

def parallel_prep(x):
    count = 0
    for sentence in x:
        prep = np.array([])
        for word in sentence:
            if word == 'as':
                word = np.array([1, 0])
            elif word == 'soon':
                word = np.array([0, 1])

            if prep.size == 0:
                prep = np.concatenate([prep, word])
            else:
                prep = np.vstack([prep, word])

        if count == 0:
            total_prep = prep.copy()
        else:
            total_prep = np.vstack([total_prep, prep])

        count = count + 1

    total_prep = np.reshape(total_prep, (-1, 3, 2))
    return total_prep

#input = serial_prep(datalist)
input_RNN = parallel_prep(datalist)
#print(input_RNN)
#print(input_RNN.shape)

target = np.array([[0],  # ["as", "soon", "as"]
                   [0],  # ["as", "soon", "as"]
                   [1]]) # ["as", "soon", "as"]
#print(target)

time_steps = input_RNN.shape[1]      #t.s(3)
sequence_length = input_RNN.shape[2] # s.l(2)
hidden_node = 3                      # h.n(3)
output_feature = target.shape[1]     #o.f(1)

Wxh = np.random.randn(sequence_length, hidden_node) # s.l(2) x h.n(3)
Whh = np.random.randn(hidden_node, hidden_node) # h.n(3) x h.n(3)
Bh = np.random.randn(1, 1)

Wy = np.random.randn(hidden_node, output_feature) # h.n(3) x o.f(1)
By = np.random.randn(1, 1)


def rnn_cell(data):
    #print('data: \n', data) # 토큰의 순서를 포함한 문장
    h = np.zeros((1, hidden_node)) # 1 x 3 (문장을 구성하는 토큰의 개수 - hidden node)
    #print('h: \n', h);
    ht_i_list = []
    ht_list = []
    ht_list.append(h)

    for sequence in data:
        #print()
        #print('sequence: ' , sequence) # 문장의 토큰 
        x = np.reshape(sequence, (1, -1))
        #print('x: ', x)

        # (1 x 1 x 2) x (2, 3)  + (1 x 3) x (3 x 3) + (1 x 1)
        h_i = np.dot(x, Wxh) + np.dot(h, Whh) + Bh
        h = np.tanh(h_i)

        ht_i_list.append(h_i)
        ht_list.append(h)

    #print('h: \n', h)
    #print('Wy: \n', Wy)
    #print('By: \n', Wy)

    pred_i = np.dot(h, Wy) + By
    pred = 1 / (1 + np.exp(-pred_i))
    return pred, pred_i, ht_list, ht_i_list,

def forward(dataset):
    count = 0
    for data in dataset:
        pred, _, _, _ = rnn_cell(data)
        if count == 0:
            total_pred = pred.copy()
        else:
            total_pred = np.concatenate([total_pred, pred], axis = 0)
        count = count + 1
    return total_pred

def BCEE_loss(y_hat, y):
    loss = np.sum(-y * np.log(y_hat) - (1-y) * np.log(1 - y_hat))
    return loss

def loss_gradient(X, Y):
    dE_dWxh_list = np.zeros_like(Wxh)
    dE_dWhh_list = np.zeros_like(Whh)
    dE_dBh_list  = np.zeros_like(Bh)

    pred, pred_i, ht_list, ht_i_list = rnn_cell(X)

    # hi_i_list [h0_i, h1_i, h2_i]
    # ht_list [h-1, h0, h1, h2]
    # x [x_0, x_1, x_2]
    loss = BCEE_loss(pred, Y)

    # Back propagation
    dE_dsig = -1 * ( (Y/pred) - ((1-Y) / (1-pred)))
    dsig_dpred_i = ( 1 / (1 + np.exp(-pred_i))) * ( 1 - 1/(1 + np.exp(-pred_i)))
    dE_dpred_i = dE_dsig * dsig_dpred_i

    # gradient of fc 
    dpred_i_dWy = np.transpose(ht_list[time_steps], (1, 0))
    dE_dWy = np.dot(dpred_i_dWy, dE_dpred_i)
    dE_dBy = dE_dpred_i

    for t in reversed(range(time_steps)):
        if t == time_steps - 1:
            dht_plus1_i_dht = np.transpose(Wy, (1, 0))
            dE_dht_plus1_i = dE_dpred_i
        else:
            dht_plus1_i_dht = np.transpose(Whh, (1, 0))
            dE_dht_plus1_i = dE_dht_i

    dE_dht = np.dot(dE_dht_plus1_i, dht_plus1_i_dht)
    dht_dht_i = (1 - np.power(np.tanh(ht_i_list[t], 2)))
    dE_dht_i = dE_dht * dht_dht_i

    #dE_dWxh
    dht_i_dWxh = np.transpose(np.reshape(X[t], (1, -1)), (1, 0))
    dE_dWxh = np.dot(dht_i_dWxh, dE_dht_i)

    #dE_dWhh
    dht_i_dWhh = np.transpose(ht_list[t], (1, 0))
    dE_dWhh = np.dot(dht_i_dWhh, dE_dht_i)

    # dE_dBh
    dE_dBh = np.sum(dE_dht_i, keepdims=True)

    dE_dWxh_list += dE_dWxh
    dE_dWhh_list += dE_dWhh
    dE_dBh_list += dE_dBh
 
    return dE_dWy, dE_dBy, dE_dWxh_list, dE_dWhh_list, dE_dBh_list

def backward(X, Y):
    dE_dWy_list = np.zeros_like(Wy)
    dE_dBy_list = np.zeros_like(By)
    dE_dWxh_list = np.zeros_like(Wxh)
    dE_dWhh_list = np.zeros_like(Whh)
    dE_dBh_list = np.zeros_like(Bh)

    for data, target in zip(X, Y):
        dE_dWy, dE_dBy, dE_dWxh, dE_dWhh, dE_dBh = loss_gradient(data, target)
        dE_dWy_list += dE_dWy
        dE_dBy_list += dE_dBy
        dE_dWxh_list += dE_dWxh
        dE_dWhh_list += dE_dWhh 
        dE_dBh_list += dE_dBh 

    return dE_dWy_list, dE_dBy_list, dE_dWxh_list, dE_dWhh_list, dE_dBh_list



pred = forward(input_RNN)
print('before pred: ', pred)

learning_rate = 0.1

for i in range(100):
    dL_dWy, dL_dBy, dL_dWxh, dL_dWhh, dL_dBh = backward(input_RNN, target)
    Wy = Wy + -1*learning_rate * dL_dWy
    By = By + -1*learning_rate * dL_dBy
    Wxh = Wxh + -1*learning_rate * dL_dWxh
    Whh = Whh + -1*learning_rate * dL_dWhh
    Bh = Bh + -1*learning_rate * dL_dBh

    if i % 10 == 0:
        pred = forward(input_RNN)
        print(i, BCEE_loss(pred, target))

