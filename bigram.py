import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
seed = 1337
batch_size = 32
block_size = 128

max_iters = 5000
eval_every = 500
eval_iters = 200

learning_rate = 3e-4
device = 'cpu'

n_emb_dims = 128
n_heads = 4
n_transformer_layers = 4
dropout_prob = 0.2
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

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B,T,C)
        q = self.query(x)   # (B,T,C)

        # compute attention scores
        weights = q @ k.transpose(-2, -1) * (C**-0.5) # (B,T,C) @ (B,C,T) -> (B,T,T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        weights = F.softmax(weights, dim=-1) # (B,T,T)
        weights = self.dropout(weights)

        # perform weighted aggregation of values
        v = self.value(x) # (B,T,C)
        out = weights @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple Heads of self-attention in parallel."""
    # Kind of like a group convolution

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_emb_dims, n_emb_dims)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """Feed Forward. Linear layer then non-linear transformation."""

    def __init__(self, n_embed_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed_dims, 4 * n_embed_dims),
            nn.ReLU(),
            nn.Linear(4 * n_embed_dims, n_embed_dims),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Intersperse communication and computation between tokens."""

    def __init__(self, n_embed_dims, n_heads):
        super().__init__()

        head_size = n_embed_dims // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embed_dims)
        self.layer_norm_1 = nn.LayerNorm(n_embed_dims)
        self.layer_norm_2 = nn.LayerNorm(n_embed_dims)

    def forward(self, x):
        """Forward pass through transformer architecture.
        Includes residual pathways and pre-norm formulation.
        """
        x = x + self.sa(self.layer_norm_1(x))     # the += makes a residual pathway to help with
        x = x + self.ffwd(self.layer_norm_2(x))   # convergence, given our deep architecture
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb_dims)
        self.position_embedding_table = nn.Embedding(block_size, n_emb_dims)

        self.blocks = nn.Sequential(*[TransformerBlock(n_emb_dims,n_heads=n_heads) 
                                      for _ in range(n_transformer_layers)])
        self.layer_norm = nn.LayerNorm(n_emb_dims)
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

        x = self.blocks(x) # apply transformer blocks (B,T,C)
        x = self.layer_norm(x) # (B,T,C)
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

    model = BigramLanguageModel(vocab_size, n_emb_dims)
    m = model.to(device)

    print(sum(p.numel() for p in m.parameters())/1e6, "M Parameters")

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
    # context = torch.tensor([encode("What! You ")], dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

    torch.save(model.state_dict(), "trained_models/macGPT.pth")

