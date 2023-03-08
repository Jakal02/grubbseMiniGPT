import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
seed = 1337
batch_size = 32
block_size = 8
max_iters = 5000
eval_every = 500
eval_iters = 200
learning_rate = 1e-3
device = 'cpu'
n_emb_dims = 32
# Reproducibility
torch.manual_seed(seed)
# ---------------

class Head(nn.Module):
    """Singular self-attention head."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb_dims, head_size, bias=False)
        self.query = nn.Linear(n_emb_dims, head_size, bias=False)
        self.value = nn.Linear(n_emb_dims, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B,T,C)
        q = self.query(x)   # (B,T,C)

        # compute attention scores
        weights = q @ k.transpose(-2, -1) * (C**-0.5) # (B,T,C) @ (B,C,T) -> (B,T,T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        weights = F.softmax(weights, dim=-1) # (B,T,T)

        # perform weighted aggregation of values
        v = self.value(x) # (B,T,C)
        out = weights @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb_dims)
        self.position_embedding_table = nn.Embedding(block_size, n_emb_dims)

        self. self_att_head = Head(n_emb_dims)
        self.lang_model_head = nn.Linear(n_emb_dims, vocab_size)

    def forward(self, idx, targets=None):
        """ Forward pass through Model.
            idx and target are (B, T) tensors of integers.
        """
        B, T = idx.shape
        # Get (B, T, C) tensor of logits
        token_embs = self.token_embedding_table(idx) # (B, T, C)
        pos_embs = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_embs + pos_embs # gets broadcasted together (B,T,C)
        x = self.self_att_head(x) # apply 1 head of self-attention. (B,T,C)
        logits = self.lang_model_head(x) # (B, T, vocab_size)
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
            # crop context to ensure its <= block_size
            idx_cond = idx[:, -block_size:]

            logits, loss = self.__call__(idx_cond) # identical to self(idx)
            # focus on only last time step
            logits = logits[:, -1, :]
            # get probabilities
            probs = F.softmax(logits, dim=-1) # -1 = figure it out pytorch

            # generate 1 character
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Data loading
def get_batch(split: str):
    """Generate batch of data of inputs x and targets y."""
    data = train_data if split == "train" else val_data

    block_inds = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in block_inds])
    y = torch.stack([data[i+1:i+block_size+1] for i in block_inds])

    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model:BigramLanguageModel):
    """Estimate loss of model."""
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

if __name__ == "__main__":
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

    # Train and Test Splits
    data = torch.tensor(encode(text), dtype=torch.long)
    num_train_chars = int(0.9*len(data))
    train_data = data[:num_train_chars]
    val_data = data[num_train_chars:]

    #-------------------

    model = BigramLanguageModel()
    m = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

    # Training Loop
    for iter in range(max_iters):

        if iter % eval_every == 0:
            losses = estimate_loss(model)
            print(f"step {iter}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}")
        
        # sample training batch
        x_batch, y_batch = get_batch('train')

        # eval loss
        logits, loss = model(x_batch, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from model
    context = torch.zeros([1,1], dtype=torch.long, device=device)
    context = torch.tensor([encode("What! You ")], dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))

