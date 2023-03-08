import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
seed = 1337
batch_size = 32
block_size = 8
max_iters = 1000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-3
device = 'cpu'
# ---------------

# Reproducibility
torch.manual_seed(seed)

# Read Text
with open('tinyshakespear.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Define Vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Define mapping between chars & ints
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[char] for char in s]
decode = lambda l: ''.join(itos[i] for i in l)

# Train and Test Splits
data = torch.tensor(encode(text), dtype=torch.long)
num_train_chars = int(0.9*len(data))
train_data = data[:num_train_chars]
val_data = data[num_train_chars:]

# Data loading
def get_batch(split: str):
    """Generate batch of data of inputs x and targets y."""
    data = train_data if split == "train" else val_data

    block_inds = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in block_inds])
    y = torch.stack([data[i+1:i+block_size+1] for i in block_inds])

    return x.to(device), y.to(device)
