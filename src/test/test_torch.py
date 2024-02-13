import torch
import numpy as np

def test_cuda_available(): 
    print("cuda" if torch.cuda.is_available() else "cpu")

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


def test_max():
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [9, 8, 7]])
    max_value, max_indices = torch.max(x, dim=1)
    print(max_value)       # 출력: tensor([3, 6, 9])
    print(max_indices)     # 출력: tensor([2, 2, 0])


#test_cuda_available()
#test_long_tensor()
#test_ones()
#test_squeeze()
#test_data_gen2()
#test_yield()
test_max()
