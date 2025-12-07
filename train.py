import torch
import sys
import yaml
import torch.optim as optim

from model import SmallGPT
from data import DataPrep

# Load the config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load the data
data_path  = config['data_path']
with open(data_path,'r') as f:
    words = f.read()

# Split into train-val set
n = int(0.9 * len(words)) # First 90% will be trained, last 10% for the val 
words_train = words[:n]
words_val = words[n:]

# Create our dataset
train_set = DataPrep(words_train)
val_set = DataPrep(words_val)

train_data = train_set.getData()
val_data = val_set.getData()
# Get batch
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-config['block_size'], (config['batch_size'],))
    x = torch.stack([data[i:i + config['block_size']] for i in ix])
    y = torch.stack([data[i+1:i + config['block_size']+1] for i in ix])
    return x,y

# Build the model
model = SmallGPT(vocab_size= train_set.vocab_size, n_emb = config['n_emb'], block_size = config['block_size'],num_heads= config['num_heads'], n_layer= config['n_layer'])


# Estimate the loss
@torch.no_grad()
def estimate_loss():
    out = {}
    # Set the data to eval mode
    model.eval()
    # Compute the loss
    for split in ['train','val']:
        losses = torch.zeros(config['eval_iters'])
        for k in range(config['eval_iters']):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        # Evaluate the mean of all eval_iters
        out[split] = losses.mean()
    # Back to train mode
    model.train()
    return out

# Set an Optimizer
optimizer = optim.AdamW(model.parameters(),lr = config['learning_rate'])




