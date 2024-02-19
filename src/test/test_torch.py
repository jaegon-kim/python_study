import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def test_cuda_available(): 
    print("cuda" if torch.cuda.is_available() else "cpu")

def test_1d_tensor():
    t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
    print(t)
    print(t.dim())
    print(t.shape)
    print(t.size())
    print(t[0], ', ', t[-1])
    print(t[2:5], ', ', t[:2])

def test_2d_tensor():
    t = torch.FloatTensor([
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.],
        [10., 11., 12.]
    ])
    print(t)
    print(t.dim())
    print(t.shape)


def test_long_tensor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src = torch.LongTensor([[1, 3, 4, 5, 6,    8, 7, 2, 9, 7]]).to(device)
    print(src, src.shape)

    data = [[1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]]
    src = torch.LongTensor(data).to(device)
    print(src, src.shape)

def test_ones():
    src_mask = torch.ones(1, 1, 10)
    print(src_mask, src_mask.shape)

def test_broad_casting():
    m1 = torch.FloatTensor([
        [3, 3]
    ])
    m2 = torch.FloatTensor([
        [2, 2]
    ])
    print(m1 + m2)

    m3 = torch.FloatTensor([
        3 # 3 -> [3, 3]
    ])
    print(m1 + m3)

    m4 = torch.FloatTensor([
        [3],
        [4]
    ]) # -> [[3, 3], [4, 4]]
    print(m1 + m4)

def test_mat_mul():
    m1 = torch.FloatTensor([
        [1, 2],
        [3, 4]
    ])
    m2 = torch.FloatTensor([
        [1],
        [2]
    ])
    # matrix multiplication
    print(m1.matmul(m2))
    # just element to element multiplication
    print(m1.mul(m2))

def test_mean():
    t = torch.FloatTensor([1, 2])
    print(t.mean())

    t = torch.FloatTensor([
        [1, 2],
        [3, 4]
    ])
    print(t.mean())
    print(t.mean(dim=0))
    print(t.mean(dim=1))
    print(t.mean(dim=-1))

def test_sum():
    t = torch.FloatTensor([
        [1, 2],
        [3, 4]
    ])
    print(t.sum())
    print(t.sum(dim=0))
    print(t.sum(dim=1))
    print(t.sum(dim=-1))
   
def test_max():
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [9, 8, 7]])
    max_value, max_indices = torch.max(x, dim=1)
    print(max_value)       # 출력: tensor([3, 6, 9])
    print(max_indices)     # 출력: tensor([2, 2, 0])


def test_tensor_manipulate():
    a = np.array([
        [
            [0, 1, 2],
            [3, 4, 5]
        ],
        [
            [6, 7, 8],
            [9, 10, 11]
        ]
    ])
    t = torch.FloatTensor(a)
    print(t)
    print(t.shape)
    t2 = t.view([-1, 3])
    print(t2)
    print(t2.shape)
    t3 = t.view([-1, 1, 3])
    print(t3)
    print(t3.shape)

    x = torch.FloatTensor([
        [1, 2],
        [3, 4]
    ])
    y = torch.FloatTensor([
        [5, 6],
        [7, 8]
    ])
    print(torch.cat([x, y], dim=0))
    print(torch.cat([x, y], dim=1))

    x = torch.FloatTensor([1, 4])
    y = torch.FloatTensor([2, 5])
    z = torch.FloatTensor([3, 6])
    print(torch.stack([x, y, z]))
    print(torch.stack([x, y, z], dim=1))

    x = torch.FloatTensor([
        [0, 1, 2],
        [2, 1, 0]
    ])
    print(torch.ones_like(x))
    print(torch.zeros_like(x))

    x = torch.FloatTensor([
        [1, 2],
        [3, 4]
    ])
    print(x.mul(2.))
    x.mul_(2.)
    print(x)


def test_squeeze():
    x = torch.tensor([1, 2, 3])
    print(x)

    # 첫 번째 차원에 새로운 차원 추가
    x_new = x.unsqueeze(0)
    print('x.unsqueeze(0): ', x_new)
    # 출력: tensor([[1, 2, 3]])

    # 두 번째 차원에 새로운 차원 추가
    x_new = x.unsqueeze(1)
    print('x.unsqueeze(1): ', x_new)

    x_new = x.unsqueeze(-2)
    print('x.unsqueeze(-2): ', x_new)


def test_type_casting():
    lt = torch.LongTensor([1, 2, 3, 4])
    print(lt)
    print(lt.float())


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        #print('  Batch.src: ', self.src)
        t = (src != pad)
        #print('  t: ', t, t.shape)
        self.src_mask = t.unsqueeze(-2)
        #print('  Batch.src_mask: ', self.src_mask, self.src_mask.shape)

        if trg is not None:
            #print('  trg         : ', trg, trg.shape)
            self.trg = trg[:, :-1]
            #print('  self.trg    : ', self.trg, self.trg.shape)
            self.trg_y = trg[:, 1:]
            #print('  self.trg_y  : ', self.trg_y, self.trg_y.shape)
            self.trg_mask = self.make_std_mask(self.trg, pad)
            #print('  self.trg_mask: ', self.trg_mask, self.trg_mask.shape)
            self.ntokens = (self.trg_y != pad).data.sum()
            #print('  self.ntokens : ', self.ntokens, self.ntokens.shape)


    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-1)
        tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


def test_data_gen2(V = 11, batch = 30, nbatches = 20):
    for i in range(nbatches):
        print('batch ', i)
        r_np = np.random.randint(1, V-1, size=(batch, 10))
        #print(' ', r_np, r_np.shape)
        data = torch.from_numpy(r_np)
        #print(' ', data, data.shape)
        data[:, 0] = 1
        #print(' ', data, data.shape)
        src = data.clone()
        tgt = data.clone()
        tgt[:, V//2:] += 1
        #print('  src         : ', src, src.shape)
        #print('  tgt         : ', tgt, tgt.shape)
        yield Batch(src, tgt, 0)
        
def test_yield():
    for i, data in enumerate(test_data_gen2()):
        print(i)
        print('  data.src  : ', data.src.shape)
        print('  data.trg  : ', data.trg.shape)
        print('  data.trg_y: ', data.trg_y.shape)




def test_autograd():
    X = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    pX = torch.pow(X, 2)

    # [4., 1., 0., 1., 4.], y = x^2 값이 나온다.
    print(pX)

    # tensor에 대해서 autograd를 할 수 없음. scalar (0차원 tensor) 값에 적용 가능한다.
    # 손실함수 들이 주로 scalar 값을 만들어 내기 때문임.
    #pX.backward()

    # tensor의 값들을 모두 더해서, scalar 값으로 만들어 준다.
    sum = torch.sum(pX)
    sum.backward()

    # [4., 1., 0., 1., 4.], y의 미분, y' = 2x 값이 나온다.
    print(X.grad) 


def linear_regression():
    x_train = torch.FloatTensor([
        [1], [2], [3]
    ])

    y_train = torch.FloatTensor([
        [2], [4], [6]
    ])

    W = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = optim.SGD([W, b], lr=0.01)

    epoches = 2000
    for epoch in range(epoches + 1):
        pred = x_train * W + b
        loss = torch.mean((pred - y_train)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('loss: ', loss.item())

    pred = x_train * W + b
    print('W   : ', W)
    print('b   : ', b)
    print('pred: ', pred)


def linear_regression_model():
    x_train = torch.FloatTensor([
        [1], [2], [3]
    ])

    y_train = torch.FloatTensor([
        [2], [4], [6]
    ])

    model = nn.Sequential(
        nn.Linear(1, 1)
    )
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epoches = 2000

    for epoch in range(epoches + 1):
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('loss: ', loss.item())

    pred = model(x_train)
    print('pred: ', pred)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)


def linear_regression_model_class():
    x_train = torch.FloatTensor([
        [1], [2], [3]
    ])

    y_train = torch.FloatTensor([
        [2], [4], [6]
    ])

    model = LinearRegressionModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epoches = 2000

    for epoch in range(epoches + 1):
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('loss: ', loss.item())

    pred = model(x_train)
    print('pred: ', pred)


def linear_regression_model_multi_dim():

    x_train = torch.FloatTensor([
                                    [73, 80, 75],
                                    [93, 88, 93],
                                    [89, 91, 90],
                                    [96, 98, 100],
                                    [73, 66, 70]
                                ])

    y_train = torch.FloatTensor([
                                    [152],
                                    [185],
                                    [180],
                                    [196],
                                    [142]
                                ])
    model = nn.Sequential(
        nn.Linear(3, 1)
    )
    #model = nn.Linear(3, 1)

    # learning rate가 너무 크면, 기울기가 발샌한다.
    # learning rate 0.0001 (1e-4)
    #optimizer = optim.SGD(model.parameters(), lr=1e-4)
    
    # learning rate 0.00001 (1e-5)
    optimizer = optim.SGD(model.parameters(), lr=1e-5)
    epoches = 2000

    for epoch in range(epoches + 1):
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('loss: ', loss.item())

    pred = model(x_train)
    print('pred: ', pred)


from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def test_mini_batch():
    x_train = torch.FloatTensor([
                                    [73, 80, 75],
                                    [93, 88, 93],
                                    [89, 91, 90],
                                    [96, 98, 100],
                                    [73, 66, 70]
                                ])

    y_train = torch.FloatTensor([
                                    [152],
                                    [185],
                                    [180],
                                    [196],
                                    [142]
                                ])
    
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = nn.Sequential(
        nn.Linear(3, 1)
    )
    # learning rate 0.00001 (1e-5)
    optimizer = optim.SGD(model.parameters(), lr=1e-5)

    nb_epoches = 20

    for epoch in range(nb_epoches + 1):
        for batch_idx, samples in enumerate(dataloader):
            #print('batch_idx: ', batch_idx)
            #print('samples: ', samples)
            x_train, y_train = samples
            pred = model(x_train)
            loss = F.mse_loss(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('loss: ', loss.item())
    new_var =  torch.FloatTensor([[73, 80, 75]]) 
    pred = model(new_var)
    print('pred: ', pred)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.x_data = [
                        [73, 80, 75],
                        [93, 88, 93],
                        [89, 91, 90],
                        [96, 98, 100],
                        [73, 66, 70]
                      ]
        self.y_data = [
                        [152],
                        [185],
                        [180],
                        [196],
                        [142]
                      ]
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y


def test_customdataset():
    dataset = CustomDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = nn.Sequential(
        nn.Linear(3, 1)
    )
    # learning rate 0.00001 (1e-5)
    optimizer = optim.SGD(model.parameters(), lr=1e-5)

    nb_epoches = 20

    for epoch in range(nb_epoches + 1):
        for batch_idx, samples in enumerate(dataloader):
            #print('batch_idx: ', batch_idx)
            #print('samples: ', samples)
            x_train, y_train = samples
            pred = model(x_train)
            loss = F.mse_loss(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('loss: ', loss.item())
    new_var =  torch.FloatTensor([[73, 80, 75]]) 
    pred = model(new_var)
    print('pred: ', pred)



torch.manual_seed(1)

#test_cuda_available()
#test_1d_tensor()
#test_2d_tensor()
#test_broad_casting()
#test_mat_mul()
#test_mean()
#test_sum()
#test_max()
#test_tensor_manipulate()
#test_squeeze()
#test_type_casting()
#test_long_tensor()
#test_ones()
#test_data_gen2()
#test_yield()
#test_autograd()
#linear_regression()
#linear_regression_model()
#linear_regression_model_class()
#linear_regression_model_multi_dim()
#test_mini_batch()
test_customdataset()
