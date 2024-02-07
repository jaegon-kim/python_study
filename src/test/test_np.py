import numpy as np 
import torch

v = np.arange(6)
print('v: ', v, ' shape: ', v.shape)

r = v.reshape(1, -1)
print('r: ', r, ' shape: ', r.shape)


for i in range(3):
    print(i)

for i in reversed(range(3)):
    print(i)

a = np.array([0, 0, 0, 0])
softmax = np.exp(a) / np.sum(np.exp(a))
print(softmax)


text = "Tokenizing is a core task of NLP"
tokenized_text = list(text)
print(tokenized_text)
s = sorted(set(tokenized_text))
print(s)
for idx, ch in  enumerate(s):
    print(idx, ':', ch)

print("cuda" if torch.cuda.is_available() else "cpu")
