import torch
import torch.nn as nn
from torch.nn import functional as F
from bigram import BigramLanguageModel

# Hyperparameters
device = 'cpu'

# Read Text
with open('tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Define Vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Define mapping between chars & ints
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[char] for char in s]
decode = lambda l: ''.join(itos[i] for i in l)


model = BigramLanguageModel(vocab_size)
model.load_state_dict(torch.load("trained_models/macGPT.pth"))

context = torch.zeros([1,1], dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
