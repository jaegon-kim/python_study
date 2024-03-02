import torch
from math import sqrt
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from transformers import AutoConfig
from transformers import AutoTokenizer
from bertviz.transformers_neuron_view import BertModel
from datasets import load_dataset


def scaled_dot_product_attention(query, key, value, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)


def test_scaled_dot_product_attention(train_data_path):
    print('* test_scaled_dot_product_attention')

    text_ids = torch.load(train_data_path + ".text_ids")
    config = torch.load(train_data_path + ".config")

    token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
    print(token_emb)

    inputs_embeds = token_emb(text_ids)
    print(inputs_embeds.size())

    query = key = value = inputs_embeds
    dim_k = key.size(-1) # 768
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    print('q(', query.size(), ') x k(', key.transpose(1, 2).size(), ') = ', scores.size() )

    r = scaled_dot_product_attention(query, key, value)
    print(r)
    print(r.shape)


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

    text_ids = torch.load(train_data_path + ".text_ids")[:100]
    print('text_ids.shape: ', text_ids.shape)
    config = torch.load(train_data_path + ".config")

    token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
    print(token_emb)

    inputs_embeds = token_emb(text_ids)
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

def store_emotion_train_data(train_data_path):
    print('* store_emotion_word_embedding')

    model_ckpt = "bert-base-uncased" # model check point
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    config = AutoConfig.from_pretrained(model_ckpt)

    config.num_labels = 6

    emotions = load_dataset("emotion", trust_remote_code=True)
    train_ds = emotions["train"]
    texts = train_ds["text"]
    labels = train_ds["label"]

    max_text_len = max(len(t) for t in texts)
    print('max_text_len: ', max_text_len)

    def text2id(t):
        return tokenizer(t, return_tensors="pt",
                         max_length = 20,
                         truncation = True,
                         padding = 'max_length',
                         add_special_tokens=False).input_ids

    text_ids = [text2id(t) for t in texts]
    text_ids = [x.squeeze() for x in text_ids]
    text_ids = torch.stack(text_ids)
    
    print('text_ids: ', text_ids.shape)
    print(text_ids[0])
    print(text_ids[1])

    labels_onehot = list(range(config.num_labels))
    labels_onehot = torch.tensor(labels_onehot)
    labels_onehot = F.one_hot(labels_onehot, num_classes=len(labels_onehot))
    labels_onehot = labels_onehot.unsqueeze(-2).to(torch.float)

    labels = [labels_onehot[x] for x in labels]
    labels = [x.squeeze() for x in labels]
    labels = torch.stack(labels)
    
    print('labels: ', labels.shape)
    print(labels[0])
    print(labels[1])

    torch.save(text_ids, train_data_path + ".text_ids")
    torch.save(labels, train_data_path + ".labels")
    torch.save(config, train_data_path + ".config")

    print("Completed to store emotion data ", train_data_path, ", size: ",
          "text_ids(", len(text_ids), ")",
          "labels(", len(labels), ")",
          "config(", 1, ")",          
          )

def validate_train_data(train_data_path):

    model_ckpt = "bert-base-uncased" # model check point
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    emotions = load_dataset("emotion", trust_remote_code=True)
    train_ds = emotions["train"]

    load_len = 10
    text_ids = torch.load(train_data_path + ".text_ids")[:load_len]
    labels = torch.load(train_data_path + ".labels")[:load_len]
    config = torch.load(train_data_path + ".config")

    def label_id2str(row):
        return train_ds.features["label"].int2str(row)
    
    def text_id2str(ids):
        return tokenizer.convert_ids_to_tokens(ids)

    for tid, label in zip(text_ids, labels):
        print("text-id: ", text_id2str(tid), ", label: ", label_id2str(label))



def train_emotion_classifier(train_data_path):
    print("* test_emotion_classification")
    
    text_ids = torch.load(train_data_path + ".text_ids")
    labels = torch.load(train_data_path + ".labels")
    config = torch.load(train_data_path + ".config")

    print("Completed to load emotion data ", train_data_path, ", size: ",
          "text_ids(", len(text_ids), ")",
          "labels(", len(labels), ")",
          "config(", 1, ")",          
          )

    model = TransformerForSequenceClassification(config)
 
    #optimizer = optim.SGD(encoder_classifier.parameters(), lr = 0.00000001)
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)

    dataset = TensorDataset(text_ids, labels)
    dataloader = DataLoader(dataset, batch_size=30, shuffle=True)

    cnt = 0
    np_epochs = 1
    
    for epoch in range(np_epochs + 1):
        for batch_idx, samples in enumerate(dataloader):
            x, y = samples
            pred = model(x)
            #loss = F.cross_entropy(pred, y)
            loss = F.kl_div(pred, y, reduction='sum')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch(', epoch, ') batch_idx(', batch_idx, ') loss: ', loss.item())

    torch.save(model, "encoder_classifier_1.model")         

train_data_path = 'hf_emotion_classifier'
#store_emotion_train_data(train_data_path)
#test_scaled_dot_product_attention(train_data_path)
#test_multi_head_attention()
#validate_train_data(train_data_path)
train_emotion_classifier(train_data_path)

