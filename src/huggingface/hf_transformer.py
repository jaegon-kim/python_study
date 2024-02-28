import torch
from math import sqrt
import torch.nn.functional as F
from torch import nn
import torch.optim as optim

from transformers import AutoConfig
from transformers import AutoTokenizer
from bertviz.transformers_neuron_view import BertModel
from datasets import load_dataset

model_ckpt = "bert-base-uncased" # model check point
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt)
text = "time flies like an arrow"

inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
print(inputs)

config = AutoConfig.from_pretrained(model_ckpt)
#print('config: ', config)

def scaled_dot_product_attention(query, key, value, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

def test_scaled_dot_product_attention(config):
    print('* test_scaled_dot_product_attention')

    token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
    print(token_emb)

    inputs_embeds = token_emb(inputs.input_ids)
    print(inputs_embeds.size())

    query = key = value = inputs_embeds
    dim_k = key.size(-1) # 768
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    print('q(', query.size(), ') x k(', key.transpose(1, 2).size(), ') = ', scores.size() )

    print(scaled_dot_product_attention(query, key, value))


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()

        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        return attn_outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        print('MultiHeadAttention: embed_dim: ', embed_dim, 'num_heads: ', num_heads, 'head_dim: ', head_dim)
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

def test_multi_head_attention():
    print('* test_multi_head_attention')

    token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
    print(token_emb)

    inputs_embeds = token_emb(inputs.input_ids)
    print(inputs_embeds.size())

    multihead_attn = MultiHeadAttention(config)
    print('inputs_embeds: ', inputs_embeds.size())
    attn_output = multihead_attn(inputs_embeds)
    print(attn_output.size())    
    feed_forward = FeedForward(config)
    ff_outputs = feed_forward(attn_output)
    ff_outputs.size()    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        hidden_state = self.layer_norm_1(x)
        x = x + self.attention(hidden_state)
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.masked_attention = MultiHeadAttention(config)
        self.cross_attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        hidden_state = self.layer_norm_1(x)
        x = x + self.masked_attention(hidden_state)
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x



class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        # print('seq_length: ', seq_length)
        # seq_length:  5
        # print('position_ids: ', position_ids)
        # position_ids:  tensor([[0, 1, 2, 3, 4]])
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config)
            for _ in range(config.num_hidden_layers) ])

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.num_labels),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        #print('input: ', x.shape)
        x = self.encoder(x)[:, 0, :]
        #print('encoder output: ', x.shape)
        x = self.dropout(x)
        #print('dropout output: ', x.shape)
        x = self.classifier(x)
        #print('clasifier output: ', x.shape)
        return x

def test_transformer_for_sequence_classification():
    print('* test_transformer_for_sequence_classification')
    config.num_labels = 6
    encoder_classifier = TransformerForSequenceClassification(config)

    text = "hello world"
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    print(inputs)
    rslt = encoder_classifier(inputs.input_ids)
    print(rslt)

def test_emotion_classification():
    print("* test_emotion_classification")

    config.num_labels = 6
    encoder_classifier = TransformerForSequenceClassification(config)

    emotions = load_dataset("emotion", trust_remote_code=True)
    train_ds = emotions["train"]

    train_set_len = 5
    texts = train_ds["text"][:train_set_len]
    labels = train_ds["label"][:train_set_len]

    def label_int2str(row):
        return train_ds.features["label"].int2str(row)

    '''
    for text, label in zip(texts, labels):
        print('\n[', label_int2str(label), '] ', text, )
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        print('inputs:\n', inputs.input_ids)
        rslt = encoder_classifier(inputs.input_ids)
        print('rslt:\n', rslt)
        print('rslt.sum:\n', torch.sum(rslt))
    '''
    labels_onehot = list(range(config.num_labels))
    labels_onehot = torch.tensor(labels_onehot)
    labels_onehot = F.one_hot(labels_onehot, num_classes=len(labels_onehot))
    labels_onehot = labels_onehot.unsqueeze(-2).to(torch.float)

    print('labels_onehot: ', labels_onehot)
    print('[0]: ', labels_onehot[0])

    #'''
    optimizer = optim.SGD(encoder_classifier.parameters(), lr = 0.00000001)
    #optimizer = optim.Adam(encoder_classifier.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)

    cnt = 0
    np_epochs = 1000
    for epoch in range(np_epochs + 1):
        for text, label in zip(texts, labels):
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            pred = encoder_classifier(inputs.input_ids)
            #print('pred: ', pred.shape, ', ', pred)
            target = labels_onehot[label]
            #print('target: ', target.shape, ', ', target)
            loss = F.cross_entropy(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            cnt += 1
            if cnt % 100:
                print('loss: ', loss.shape,' ', loss)
    #'''
#test_scaled_dot_product_attention(config)
#test_multi_head_attention()
#test_transformer_for_sequence_classification()
test_emotion_classification()

