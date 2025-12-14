from layers import Block

import torch
import torch.nn as nn
from torch.nn import functional as F

class SmallGPT(nn.Module):
    def __init__(self, vocab_size, n_emb, block_size, num_heads, n_layer,dropout):
        super().__init__()
        self.block_size = block_size
        self.token_emb_table = nn.Embedding(vocab_size,n_emb)
        self.position_emb_table = nn.Embedding(block_size,n_emb)
        self.blocks = nn.Sequential(*[Block(num_heads=num_heads, n_emb =n_emb, block_size=block_size,dropout= dropout) for _ in range(n_layer)])
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
        pos_emb = self.position_emb_table(torch.arange(T))

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
    def save_checkpoint(self, filepath, optimizer=None, step=0, train_loss=None, val_loss=None, config=None, vocab_size=None):
        checkpoint = {
            'step': step,
            'model_state_dict': self.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config,
            'vocab_size': vocab_size,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    @staticmethod
    def load_checkpoint(filepath, model, optimizer=None):
        print(f"Loading checkpoint from {filepath}")
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        info = {
            'step': checkpoint.get('step', 0),
            'train_loss': checkpoint.get('train_loss', None),
            'val_loss': checkpoint.get('val_loss', None),
            'config': checkpoint.get('config', None),
            'vocab_size': checkpoint.get('vocab_size', None),
        }
        
        print(f"Resumed from step {info['step']}, train loss: {info['train_loss']:.4f}, val loss: {info['val_loss']:.4f}")
        return info
    def generate(self, idx, max_new_tokens):
        # id (B,T)
        for _ in range(max_new_tokens):
            # Last block_size characters
            idx_cap = idx[:, -self.block_size:]
            # Get the logits for the next one
            logits, loss = self(idx_cap) # B,C
            # We only care about the last time step
            logits = logits[:,-1,:]
            # Get the probs & sample
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples= 1) # (B,1)
            idx = torch.cat((idx,idx_next), dim = 1) # (B,T+1)

        return idx



    