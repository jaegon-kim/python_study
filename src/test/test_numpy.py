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

def test_transpose():
    X = np.array([[1, 2, 3]])
    W = np.array([[3],
                [2],
                [1]])
    print('X: ', X.shape, '\n', X)
    print('W: ', W.shape, '\n', W)
    print()

    # axes 파라미터는 차원의 순서를 타나낸다.
    # 0차원이 1차원이 되고, 1차원이 0차원이 되었다.
    Xt = np.transpose(X, axes=(1, 0))
    Wt = np.transpose(W, axes=(1, 0))
    print('Xt: ', Xt.shape, '\n', Xt)
    print('Wt: ', Wt.shape, '\n', Wt)
    print()

    # axes 파라미터로 차원의 순서를 그대로 하면, 원래의 행렬이 출력된다.
    Xt = np.transpose(X, axes=(0, 1))
    Wt = np.transpose(W, axes=(0, 1))
    print('Xt: ', Xt.shape, '\n', Xt)
    print('Wt: ', Wt.shape, '\n', Wt)
    print()

def test_data_split():
    np_array = np.arange(0, 16).reshape((4, 4))
    print('np_array:\n', np_array)
    x = np_array[:, 3]
    print('np_array[:, 3]\n', x)

    y = np_array[:, :3]
    print('np_array[:, 3]\n', y)



test_reshape()
test_softmax_of_zero_weight()
test_transpose()
test_data_split()



