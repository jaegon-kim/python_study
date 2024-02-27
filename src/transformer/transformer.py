# Reference
# https://colab.research.google.com/drive/1oOTy4z1IQt0PXmox65JADucwQY6V6NsW

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt

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


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        # 4개의 Linear 계층이 Clone 된다.
        # Linear 0: query 용
        # Linear 1: key 용
        # Linear 2: value 용
        # Linear 3: attention 용 
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1)
        # [greedy_decode case]
        # 
        #  l = self.linears[0]  // nn.Linear(512, 512) 
        #  x = query            // shape: [1, 10, 512]
        #  x = nn.Linear(x)   // 선형층 통과 (W_q 값은 어디서 왔을까 ? Training에서 학습 한 값 인가 ?)
        #  x = out.view(nbatches = 1, -1 = 10, self.h = 8, self.d_k = 64) // shape: [1, 10, 8, 64]
        #  x = out.transpose(1, 2) // // shape: [1, 10, 8, 64]
        #  query = x
        #  
        #  위 과정을 (l = self.linears[1], x = key), (l = self.linears[2], x = key)에 대해 반복하여
        #  query, key, value를 만든다.

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                for l, x in zip(self.linears, (query, key, value))]
        # 2)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 4)
        return self.linears[-1](x)


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
    
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 주의 !! 그냥 ,dropout 이 아닌 원래 word vector와 Add 함
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class PositionwiseForwards(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseForwards, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn # cross attention
        self.feed_forward = feed_forward # positional feed forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # memory: 인코더로 부터 넘어온 인코딩 (nbatches, n_seq, d_model)
        m = memory

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # cross attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x
    
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
    
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        #print('  forward src: ', src.shape, ', mask: ', src_mask.shape)
        #print('  forward tgt: ', tgt.shape, ', mask: ', tgt_mask.shape)
        r = self.decode(
            self.encode(src, src_mask), src_mask, tgt, tgt_mask
        )
        #print('  forward r  : ', r.shape)
        return r

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseForwards(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        # Encoder
        # 1 attention layer + 1 feed forward layer
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        # Decoder
        # 2 attention layer + 1 feed forward layer
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        # Src Embeding
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        # Target Embeding
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),

        Generator(d_model, tgt_vocab)
    )
    #print('*model: \n', model)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    # To use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model

#tmp_model = make_model(10, 10, 2)

class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            # src: 1 5 3 7 1 4 4 6 8 3 (len: 10)
            # trg: 1 5 3 7 1 5 5 7 8 4 (len: 10)
            #    trg: 1 5 3 7 1 5 5 7 8 (len: 9) - 디코더에 입력되는 Target
            #  trg_y: 5 3 7 1 5 5 7 8 4 (len: 9) - 디코더에서 출력되는 Target

            # 디코더에 입력되는 Target
            self.trg = trg[:, :-1]
            # 디코더에서 출력되는 Target
            self.trg_y = trg[:, 1:]

            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-1)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask
    
def data_gen2(V, batch, nbatches):

    # To use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(nbatches):
        # To use GPU
        #data = torch.from_numpy(np.random.randint(1, V-1, size=(batch, 10)))
        data = torch.from_numpy(np.random.randint(1, V-1, size=(batch, 10))).to(device)
        data[:, 0] = 1
        src = data.clone()
        tgt = data.clone()
        tgt[:, (V//2):] += 1
        #tgt[:, 5] -= 1

        yield Batch(src, tgt, 0)
'''
for data in data_gen2(11, 1, 10):
    print('data.src: ', data.src)
    #print('data.src_mask: ', data.src_mask)
    print('data.trg: ', data.trg)
    print('data.trg_y: ', data.trg_y)
    #print('data.trg_mask\n', data.trg_mask)
    print()
'''
    
global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def get_std_opt(model):
        return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
'''
opts = [NoamOpt(512, 1, 4000, None),
        NoamOpt(512, 1, 8000, None),
        NoamOpt(256, 1, 4000, None)]
plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])
'''

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()

        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        true_dist[:, self.padding_idx] = 0

        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        return self.criterion(x, true_dist)

class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                            y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss * norm
    
def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d, Loss: %f, Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens

def training_transformer():

    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    #model = make_model(V, V, N=6) # 6번이 더 안 좋은 듯 .. 왜 ?

    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 1200,
                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    epoches = 19;
    #epoches = 1;

    for epoch in range(epoches + 1):
        # model goes into training mode
        model.train()
        print(f'{epoch} epoch train')

        # 'data_gen2(V, 30, 20)' will give random (30, 10) data 20 times
        run_epoch(data_gen2(V, 30, 20), model,
                SimpleLossCompute(model.generator, criterion, model_opt))
        print('eval')
        # model goes out from training mode, and goes into evaluation mode
        model.eval()
        eval_loss = run_epoch(data_gen2(V, 30, 5), model,
                SimpleLossCompute(model.generator, criterion, None))
        print('eval_loss:', eval_loss, '\n')

    return model


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    #print('ys: ', ys)

    for i in range(max_len-1):
        #print('ys.shape: ', ys.shape)
        out = model.decode(
            memory, src_mask, ys,
            subsequent_mask(ys.size(1)).type_as(src.data)
            )
        #print('out.shape: ', out.shape)
        #print('out[:, -1].shape: ', out[:, -1].shape)

        prob = model.generator(out[:, -1])
        #print('prob: ', prob)
        #print('prob.shape:', prob.shape)

        _, next_word = torch.max(prob, dim = 1)
        #print('next_word: ', next_word, ', next_word.data[0]: ', next_word.data[0])
        next_word = next_word.data[0]

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        #print('ys: ', ys)
        #print('\n')
    return ys

def eval_transformer(model):

    model.eval()

    # To use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #src = torch.LongTensor([[6, 7, 8, 9, 5,     1, 2, 3, 4, 5]]).to(device)
    #src = torch.LongTensor([[1, 2, 3, 4, 5,    6, 7, 8, 9, 5]]).to(device)
    src = torch.LongTensor([[1, 3, 4, 5, 6,    8, 7, 2, 9, 7]]).to(device)
                           # 1, 3, 4, 5, 6,    9, 8, 3, 10, 8
    src_mask = torch.ones(1, 1, 10).to(device)
 
    print('input : ', src)
    print('output: ', greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))


mode_path = 'my_transformer_1.model'

model = training_transformer()
torch.save(model, mode_path)

#model = torch.load(mode_path)
#eval_transformer(model)

