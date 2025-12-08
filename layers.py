import torch

# Now we use nn.Module
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self,head_size,n_emb, block_size,dropout):
        super().__init__()
        # Initializing the weight matrices of query, key, value 
        self.q = nn.Linear(n_emb,head_size,bias = False)
        self.k = nn.Linear(n_emb,head_size,bias = False)
        self.v = nn.Linear(n_emb,head_size,bias = False)

        # Register buffer for the tril, as it is not really a part of a model
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        B,T,C = x.shape
        # Calculate the query and the keys
        k = self.k(x) 
        q = self.q(x)

        # Measure the dot product, divide all by the dimension
        wei = q @ k.transpose(-2,-1) * k.shape[-1] ** -0.5

        # Masking
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei,dim = -1)
        # Add the Dropout here to prevent some communication
        wei = self.dropout(wei)
        # Use the values
        v = self.v(x)
        out  = wei @ v 
        return out
class MultiHead(nn.Module):
    def __init__(self,num_heads, head_size, n_emb, block_size,dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size,n_emb,block_size,dropout) for _ in range(num_heads)])

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim = -1)
        return out
class FeedForward(nn.Module):
    def __init__(self, n_emb,dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb,n_emb),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        out = self.net(x)
        return out
class Block(nn.Module):
    # Transformer Block
    def __init__(self,num_heads, n_emb, block_size,dropout):
        super().__init__()
        head_size = n_emb//num_heads
        self.sa = MultiHead(num_heads, head_size, n_emb, block_size,dropout)
        self.ffwd = FeedForward(n_emb,dropout)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self,x):
        # Residual connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
'''# This is how I will implement the LayerNorm
class LayerNorm(nn.Module):
    def __init__(self,dim,eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gain = torch.ones(dim)
        self.bias = torch.zeros(dim)
    def forward(self,x):
        mean = torch.mean(x,1,keepdim = True)
        var = torch.var(x,1,keepdim = True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gain * x_hat + self.bias
        return out
    def parameters(self):
        return [self.gain,self.bias]
'''
        


