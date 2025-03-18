import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, copy, time
import matplotlib.pyplot as plt


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

    x1 = torch.randint(-1, 1, (1, 10, 512), requires_grad=False)
    print('x1:', x1.shape)
    x2 = x1.view(1, -1, 8, 64)
    print('x2:', x2.shape)
    x3 = x2.transpose(1, 2)
    print('x3:', x3.shape)
    x4 = x3.transpose(-2, -1)
    print('x4:', x4.shape)

    x5 = torch.matmul(x3, x4) / math.sqrt(x3.size(-1))
    print('x5:', x5.shape)

    x6 = x3.transpose(1, 2)
    print('x6:', x6.shape)

    x7 = x6.contiguous()
    print('x7:', x6.shape)


def test_masked_fill():
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.tensor([[True, False, True], [False, True, False]])
    new_tensor = tensor.masked_fill(mask, 0)
    print(new_tensor)

    mask = torch.tensor([[True, False, True]])
    new_tensor = tensor.masked_fill(mask, 0)
    print(new_tensor)


def test_squeeze():
    x = torch.tensor([1, 2, 3])
    print(x, x.shape)

    # 첫 번째 차원에 새로운 차원 추가
    x_new = x.unsqueeze(0)
    print('x.unsqueeze(0): ', x_new, x_new.shape)

    x_old = x.squeeze()
    print('x.squeeze(): ', x_old, x_old.shape)


    # 출력: tensor([[1, 2, 3]])

    # 두 번째 차원에 새로운 차원 추가
    x_new = x.unsqueeze(1)
    print('x.unsqueeze(1): ', x_new, x_new.shape)

    x_new = x.unsqueeze(-2)
    print('x.unsqueeze(-2): ', x_new, x_new.shape)


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
        print(' ', r_np.shape)
        data = torch.from_numpy(r_np)
        #print(' ', data, data.shape)
        data[:, 0] = 1
        #print(' ', data, data.shape)
        src = data.clone()
        tgt = data.clone()
        tgt[:, V//2:] += 1
        #print('  src         : ', src, src.shape)
        #print('  tgt         : ', tgt, tgt.shape)
        #yield Batch(src, tgt, 0)
        
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


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(
            self.linear(x)
        )


def logistics_regression_model_class():
    x_data = [
        [1, 2], 
        [2, 3], 
        [3, 1], 
        [4, 3], 
        [5, 3], 
        [6, 2]
    ]
    y_data = [
        [0], 
        [0], 
        [0], 
        [1], 
        [1], 
        [1]
    ]
    x_train = torch.FloatTensor(x_data)
    y_train = torch.FloatTensor(y_data)

    model = BinaryClassifier()
    optimizer = optim.SGD(model.parameters(), lr = 1)

    np_epochs = 1000
    for epoch in range(np_epochs + 1):
        pred = model(x_train)
        loss = F.binary_cross_entropy(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('loss: ', loss.item())

    print(model(x_train))

torch.manual_seed(1)

def test_embedding():
    vocab_size = 11
    embedding_dim = 512
    embedding_layer = nn.Embedding(vocab_size, embedding_dim)

    input_indices = torch.tensor([[1, 3, 4, 5, 6,    8, 7, 2, 9, 7]])
    embeddings = embedding_layer(input_indices)
    print(input_indices.shape)
    print(embeddings)
    print(embeddings.shape)

def test_embedding_with_positional_enconding():

    class Embeddings(nn.Module):
        def __init__(self, d_model, vocab_size): # d_model = 512, vocab_size = 11로 초기화
            super(Embeddings, self).__init__()
            self.lut = nn.Embedding(vocab_size, d_model)
            self.d_model = d_model

        def forward(self, x):
            # lut(x) : Word Embedding (torch.Size([1, 10]) -> torch.Size([1, 10, 512]))
            # 제곱근 (sqrt)를 곱하여 Position Embedding에 의해 워드 임베딩이 희석되는 것을 줄인다.
            return self.lut(x) * math.sqrt(self.d_model)
        
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) *
                                -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x):
            pe_val = self.pe[:, :x.size(1)]
            pe_val.requires_grad = False
            x = x + pe_val
            return self.dropout(x)

    torch.random.manual_seed(15)

    # 단어장 크기 11, 임베딩 벡터 길이 10인 임베딩 층 선언
    V = 11
    d_model = 10
    emb = Embeddings(d_model, V)
    # 길이 10인 문장의 위치를 인코딩하는 위치 인코딩 층 선언
    pe  = PositionalEncoding(d_model, 0.1)
    # 그림 그리기 위해 positional encoding을 복사해둠
    pe_ = pe.pe.clone()

    # 문장 길이는 5이고
    n_seq = 5
    # 0~10까지 숫자 5개를 무작위로 뽑아서 문장을 구성
    # 이 예에서 문장은 실제 단어로 구성된 것은 아니고 0, 1, 2, 3, ..., 10인 
    # 숫자를 단어로 간주함
    x = torch.randint(0, V-1, (1, n_seq,), requires_grad=False)
    print("integer tokens:", x.shape, "\n", x)

    # Feed forward Embeddings-PositionalEncoding
    # 숫자(단어) 다섯개로 구성된 입력을 임베딩층을 통해 (n_seq, d_model)로 변환
    embedded = emb(x)
    print("embedded:", embedded.shape, "\n", embedded)

    # 임베딩 벡터에 대한 위치 인코딩 정보를 구해서 임베딩 벡터에 더함
    embedded_pe = pe(embedded)
    print("embedded_pe:", embedded_pe.shape, "\n", embedded_pe)

    # 임베딩 벡터와 위치 인코딩이 더해진 입력 벡터를 그림 
    fig, axs = plt.subplots(figsize=(18,3), nrows=1, ncols=3)
    axs[0].imshow(embedded.detach().numpy()[0], aspect='auto', cmap='gray')
    axs[0].set_xlabel('d_model')
    axs[0].set_ylabel('n_seq')
    axs[0].set_title(f"embedded vector x")

    axs[1].imshow(pe_.numpy()[0][:x.shape[1]], aspect='auto', cmap='gray')
    axs[1].set_xlabel('d_model')
    axs[1].set_ylabel('n_seq')
    axs[1].set_title(f"positional encoding")

    axs[2].imshow(embedded_pe.detach().numpy()[0], aspect='auto', cmap='gray')
    axs[2].set_xlabel('d_model')
    axs[2].set_ylabel('n_seq')
    axs[2].set_title(f"embedded vector x + positional encoding")

    plt.show()

def test_dropout():
    # 드롭아웃 비율 설정 (여기서는 0.3로 설정)
    # 3/10이 0이 될것이다.
    dropout_rate = 0.3

    # 드롭아웃 레이어 생성
    dropout = nn.Dropout(p=dropout_rate)

    # 입력 데이터 생성 (예: 크기가 3인 텐서)
    input_tensor = torch.randn(10)

    # 드롭아웃 레이어 적용
    output_tensor = dropout(input_tensor)

    # 드롭아웃이 적용된 출력 확인
    print("Input Tensor:", input_tensor)
    print("Output Tensor after dropout:", output_tensor)

def test_layer_norm():

    class LayerNorm(nn.Module):
        def __init__(self, features, eps=1e-6):
            super(LayerNorm, self).__init__()
            self.a_2 = nn.Parameter(torch.ones(features))
            self.b_2 = nn.Parameter(torch.zeros(features))
            self.eps = eps

        def forward(self, x):
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, unbiased=False, keepdim=True)
            return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

    layer_norm = LayerNorm(5)

    #input_tensor = torch.randn(5)
    input_tensor = torch.FloatTensor([
                                [1, 3, 5, 7, 9]
                        ])

    output_tensor = layer_norm(input_tensor)
    print("Input Tensor:", input_tensor)
    print("Output Tensor after layer normalization:", output_tensor)

    mean = output_tensor.mean(-1, keepdim=True)
    std = output_tensor.std(-1, unbiased=False, keepdim=True)
    print("mean of Output Tensor: ", mean)
    print("std of Output Tensor: ", std)

def test_attention():

    def clones(module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    class MultiHeadAttention(nn.Module):
        def __init__(self, h, d_model, dropout=0.1):
            super(MultiHeadAttention, self).__init__()
            assert d_model % h == 0
            self.d_k = d_model // h
            self.h = h
            self.linears = clones(nn.Linear(d_model, d_model), 4)
            self.attn = None
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, query, key, value, mask=None):
            if mask is not None:
                mask = mask.unsqueeze(1)
            nbatches = query.size(0)

            # for l, x in zip(self.linears, (query, key, value)):
            #     print('\nl: ', l, ', x: \n', x)
            #     v = l(x).view(nbatches, -1, self.h, self.d_k)
            #     print('v.transpose(1, 2):\n', v.transpose(1, 2))

            l = self.linears[0]
            x = query
            print('\nl: ', l, ', x: \n', x)
            v = l(x).view(nbatches, -1, self.h, self.d_k)
            print('v.transpose(1, 2):\n', v.transpose(1, 2))
            query = x

            # 1)
            #print('query: ', query.shape)
            #query, key, value = \
            #    [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            #        for l, x in zip(self.linears, (query, key, value))]
            #print('query: ', query.shape)

            #print('query: ', query)
            #print('key: ', key)
            #print('value: ', value)


            # 2)
            #x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
            # 3)
            #x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
            # 4)
            #return self.linears[-1](x)

    x = torch.tensor([[
        [-1.3702,  1.0295,  0.4944, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2364, -1.4332,  3.1860],
        [-1.1482,  1.2761,  2.5505, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  2.3037,  1.0941, -0.1095],
        [ 0.7402,  0.9188,  0.5695, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0.9548,  2.2250,  3.1030],
         
        [-0.4645, -1.0368,  0.3478, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.8372,  0.7474,  2.4098],
        [ 0.4066,  0.9719,  1.2753, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6630,  1.2791, -0.8821],
        [ 1.3251,  1.5064,  2.4238, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1342,  2.0200,  2.0517]
        ]])

    print('x: \n', x)

    mask = torch.tensor([[
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    ]])

    m_head_attention = MultiHeadAttention(8, 16)
    m_head_attention(x, x, x, mask)

def test_softmax():
    x = torch.tensor([
        [8., 4., 8.],
        [4., 2., 4.],
        [5., 2., 6.]
    ])
    p = F.softmax(x, dim=-1)
    print(p)

torch.set_printoptions(sci_mode=False, precision=1)

def test_onehot_encoding():
    #input_ids = [0, 1, 2, 3, 4, 5]
    input_ids = list(range(6))
    input_ids = torch.tensor(input_ids)
    one_hot_encodings = F.one_hot(input_ids, num_classes=len(input_ids))
    one_hot_encodings = one_hot_encodings.unsqueeze(-2)
    print(one_hot_encodings)
    print('[0]: ', one_hot_encodings[0])

test_cuda_available()
#test_1d_tensor()
#test_2d_tensor()
#test_broad_casting()
#test_mat_mul()
#test_mean()
#test_sum()
#test_max()
#test_tensor_manipulate()
#test_masked_fill()
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
#test_customdataset()
#logistics_regression_model_class()
#test_embedding()
#test_embedding_with_positional_enconding()
#test_dropout()
#test_layer_norm()
#test_attention()
#test_softmax()
#test_onehot_encoding()