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

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        """ Forward pass through Model.
            idx and target are (B, T) tensors of integers.
        """
        # Get (B, T, C) tensor of logits
        logits = self.token_embedding_table(idx)
        loss = None

        if targets is not None:
            B, T, C = logits.shape
            # Re-shape to feed into cross_entropy
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """ Generate new text from context idx.
            idx is (B, T) array of indices in current context.
        """
        for _ in range(max_new_tokens):
            logits, loss = self.__call__(idx) # identical to self(idx)
            # focus on only last time step
            logits = logits[:, -1, :]
            # get probabilities
            probs = F.softmax(logits, dim=-1) # -1 = figure it out pytorch
            
            # generate 1 character
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

