from layers import Block

import torch
import torch.nn as nn
from torch.nn import functional as F

class SmallGPT(nn.Module):
    def __init__(self, vocab_size, n_emb, block_size, num_heads, n_layer):

        self.token_emb_table = nn.Embedding(vocab_size,n_emb)
        self.position_emb_table = nn.Embedding(block_size,n_emb)
        self.blocks = nn.Sequential(*[Block(num_heads=num_heads, n_emb =n_emb, block_size=block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)

        self.apply(self._init_weights)
    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.normal_(module.weight,mean = 0.0, std = 0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
    def forward(self,idx, targets = None):
        B,T = idx.shape
        
        # idx (B,T)
        tok_emb = self.token_emb_table(idx) 
        pos_emb = self.position_emb_table(torch.arrange(T))

        x = tok_emb + pos_emb
        # Blocks + LayerNorm + MLP 
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss




    