import numpy as np 

def test_reshape():
    v = np.arange(6)
    print('v: ', v, ' shape: ', v.shape)

    r = v.reshape(1, -1)
    print('r: ', r, ' shape: ', r.shape)

    r = v.reshape(1, 1, -1)
    print('r: ', r, ' shape: ', r.shape)

    r = v.reshape(1, 1, 1, -1)
    print('r: ', r, ' shape: ', r.shape)


def test_softmax_of_zero_weight():
    a = np.array([0, 0, 0, 0])
    softmax = np.exp(a) / np.sum(np.exp(a))
    print(softmax)


test_reshape()
test_softmax_of_zero_weight()

