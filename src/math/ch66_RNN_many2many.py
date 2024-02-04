import numpy as np
import sys

np.random.seed(230912)

dict_hello = {'h':np.array([1, 0, 0, 0]), \
              'e':np.array([0, 1, 0, 0]), \
              'l':np.array([0, 0, 1, 0]), \
              'o':np.array([0, 0, 0, 1]) \
            }

print(dict_hello)

inputs = np.array([])

for text in "hell":
    #print(text)
    for key, value in dict_hello.items():
        if key == text:
            #print(key, '-', value)
            if inputs.size == 0:
                inputs = np.concatenate([inputs, value])
            else:
                inputs = np.vstack([inputs, value])

print('inputs: \n', inputs)
input_RNN = np.reshape(inputs, (1, 4, -1))
print('input_RNN: \n', input_RNN)
print('shape: ', input_RNN.shape)


target = np.array([])

for text in 'ello':
    for key, value in dict_hello.items():
        if key == text:
            if target.size == 0:
                target = np.concatenate([target, value])
            else:
                target = np.vstack([target, value])

print('target :\n', target)
targets = np.reshape(target, (1, 4, -1))
print('targets: /n', targets)
print('shape: ', targets.shape)

#sys.exit()

time_steps = input_RNN.shape[1]      #t.s(4)
sequence_length = input_RNN.shape[2] # s.l(4)
hidden_node = 3                      # h.n(3)
output_feature = target.shape[1]     #o.f(4)

Wxh = np.random.randn(sequence_length, hidden_node) # s.l(2) x h.n(3)
Whh = np.random.randn(hidden_node, hidden_node) # h.n(3) x h.n(3)
Bh = np.random.randn(1, 1)

Wy = np.random.randn(hidden_node, output_feature) # h.n(3) x o.f(4)
By = np.random.randn(1, 1)

def rnn_cell(data):
    h = np.zeros((1, hidden_node)) # 1 x 3 (문장을 구성하는 토큰의 개수 - hidden node)
    #print('h: \n', h);
    ht_i_list = []
    ht_list = []
    pred_i_list = []
    pred_list = []
    ht_list.append(h)
    count = 0

    #print('rnn_cell')
    #print('data: ', data)
    for sequence in data:
        x = np.reshape(sequence, (1, -1))
        #print('x: ', x)

        # (1 x 1 x 2) x (2, 3)  + (1 x 3) x (3 x 3) + (1 x 1)
        h_i = np.dot(x, Wxh) + np.dot(h, Whh) + Bh

        h = np.tanh(h_i)
        #print(' h_i: ', h_i)
        #print(' h  : ', h)

        pred_i = np.dot(h, Wy) + By
        pred = np.exp(pred_i)/np.sum(np.exp(pred_i))

        #print(' pred_i: ', pred_i)
        #print(' pred  : ', pred)

        ht_i_list.append(h_i)
        ht_list.append(h)
        pred_i_list.append(pred_i)
        pred_list.append(pred)

        if count == 0:
            total_pred = pred.copy()
        else:
            total_pred = np.concatenate([total_pred, pred], axis = 0)
        count = count + 1   
    return total_pred, pred_list, pred_i_list, ht_list, ht_i_list


predics, pred_list, pred_i_list, ht_list, ht_i_list = rnn_cell(input_RNN[0])

# sys.exit()

print('predics: \n', predics)
print('pred_list: \n', pred_list)
print('pred_i_list: \n', pred_i_list)
print('ht_list: \n', ht_list)
print('ht_list: \n', ht_i_list)



def CEE_loss(y_hat, y):
    loss = np.sum(-y * np.log(y_hat))
    return loss

losses = CEE_loss(predics, targets[0])
print("losses: \n", losses)

'''
def loss_gradient(X, Y):
    dE_dWxh_list = np.zeros_like(Wxh)
    dE_dWhh_list = np.zeros_like(Whh)
    dE_dBh_list  = np.zeros_like(Bh)
    dE_dWy_list  = np.zeros_like(Wy)
    dE_dBy_list  = np.zeros_like(By)

    predics, pred_list, pred_i_list, ht_list, ht_i_list = rnn_cell(X)

    loss = CEE_loss(predics, Y)
    count = 0

    for t_s in reversed(range(time_steps)):
        dE_dpred_i = pred_list[t_s] - Y[t_s]

        dpred_i_dWy = np.transpose(ht_list[t_s + 1], (1, 0))
        dE_dWy = np.dot(dpred_i_dWy, dE_dpred_i)
        dE_dBy = np.sum(dE_dpred_i, keepdims=True)
        dE_dWy_list += dE_dWy
        dE_dBy_list += dE_dBy

        for i in reversed(range(t_s +1)):
            if i == t_s:
                dht_plus1_i_dht = np.transpose(Wy, (1, 0))
                dE_dht_plus1_i = dE_dpred_i
            else:
                dht_plus1_i_dht = np.transpose(Whh, (1, 0))
                dE_dht_plus1_i = dE_dht_i

            dE_dht = np.dot(dE_dht_plus1_i, dht_plus1_i_dht)
            dht_dht_i = (1-np.power(np.tanh(ht_i_list[i]), 2))
            dE_dht_i = dE_dht * dht_dht_i

            # de_dWxh
            dht_i_dWxh = np.transpose(np.reshape(X[i], (1, -1)), (1, 0))
            dE_dWxh = np.dot(dht_i_dWxh, dE_dht_i)

            # dE_dWhh
            dht_i_dWhh = np.transpose(ht_list[i], (1, 0))
            dE_dWhh = np.dot(dht_i_dWhh, dE_dht_i)

            # dE_dBh
            dE_dBh = np.sum(dE_dht_i, keepdims=True)

            dE_dWxh_list += dE_dWxh
            dE_dWhh_list += dE_dWhh
            dE_dBh_list += dE_dBh

    return dE_dWy_list, dE_dBy_list, dE_dWxh_list, dE_dWhh_list, dE_dBh_list
'''

def loss_gradient_tbptt(X, Y, lr):
    global Wy, By, Wxh, Whh, Bh

    dE_dWxh_list = np.zeros_like(Wxh)
    dE_dWhh_list = np.zeros_like(Whh)
    dE_dBh_list  = np.zeros_like(Bh)

    #dE_dWy_list  = np.zeros_like(Wy)
    #dE_dBy_list  = np.zeros_like(By)

    predics, pred_list, pred_i_list, ht_list, ht_i_list = rnn_cell(X)

    loss = CEE_loss(predics, Y)
    count = 0

    for t_s in reversed(range(time_steps)):
        dE_dpred_i = pred_list[t_s] - Y[t_s]

        dpred_i_dWy = np.transpose(ht_list[t_s + 1], (1, 0))
        dE_dWy = np.dot(dpred_i_dWy, dE_dpred_i)
        dE_dBy = np.sum(dE_dpred_i, keepdims=True)

        #dE_dWy_list += dE_dWy
        #dE_dBy_list += dE_dBy

        for i in reversed(range(t_s +1)):
            if i == t_s:
                dht_plus1_i_dht = np.transpose(Wy, (1, 0))
                dE_dht_plus1_i = dE_dpred_i
            else:
                dht_plus1_i_dht = np.transpose(Whh, (1, 0))
                dE_dht_plus1_i = dE_dht_i

            dE_dht = np.dot(dE_dht_plus1_i, dht_plus1_i_dht)
            dht_dht_i = (1-np.power(np.tanh(ht_i_list[i]), 2))
            dE_dht_i = dE_dht * dht_dht_i

            # de_dWxh
            dht_i_dWxh = np.transpose(np.reshape(X[i], (1, -1)), (1, 0))
            dE_dWxh = np.dot(dht_i_dWxh, dE_dht_i)

            # dE_dWhh
            dht_i_dWhh = np.transpose(ht_list[i], (1, 0))
            dE_dWhh = np.dot(dht_i_dWhh, dE_dht_i)

            # dE_dBh
            dE_dBh = np.sum(dE_dht_i, keepdims=True)

            dE_dWxh_list += dE_dWxh
            dE_dWhh_list += dE_dWhh
            dE_dBh_list += dE_dBh

        Wy = Wy + -1 * lr * dE_dWy
        By = By + -1 * lr * dE_dBy
        Wxh = Wxh + -1 * lr * dE_dWxh_list
        Whh = Whh + -1 * lr * dE_dWhh_list
        Bh = Bh + -1 * lr * dE_dBh_list

   #return dE_dWy_list, dE_dBy_list, dE_dWxh_list, dE_dWhh_list, dE_dBh_list

'''
print(loss_gradient(input_RNN[0], targets[0]))

pred, _, _, _, _ = rnn_cell(input_RNN[0])
print('before pred: ', pred)


learning_rate = 0.005

print('input_RNN[0]: \n', input_RNN[0])

for i in range(2000):
    dL_dWy, dL_dBy, dL_dWxh, dL_dWhh, dL_dBh = loss_gradient(input_RNN[0], target[0])
    Wy = Wy + -1*learning_rate * dL_dWy
    By = By + -1*learning_rate * dL_dBy
    Wxh = Wxh + -1*learning_rate * dL_dWxh
    Whh = Whh + -1*learning_rate * dL_dWhh
    Bh = Bh + -1*learning_rate * dL_dBh

    if i % 100 == 0:
        pred, _, _, _, _ = rnn_cell(input_RNN[0])
        print(i, CEE_loss(pred, target[0]))

pred, _, _, _, _ = rnn_cell(input_RNN[0])
print('after pred: ', pred)
'''
#print(loss_gradient_tbptt(input_RNN[0], targets[0], lr=0.001))

pred, _, _, _, _ = rnn_cell(input_RNN[0])
print('before pred: ', pred)


#print('input_RNN[0]: \n', input_RNN[0])

for i in range(2000):
    loss_gradient_tbptt(input_RNN[0], targets[0], lr=0.005)

    if i % 100 == 0:
        pred, _, _, _, _ = rnn_cell(input_RNN[0])
        print(i, CEE_loss(pred, target[0]))

pred, _, _, _, _ = rnn_cell(input_RNN[0])
print('after pred: ', pred)