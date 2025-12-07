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
train_data = DataPrep(words_train).getData()
val_data = DataPrep(words_val).getData()

# Get batch
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-config['block_size'], (config['batch_size'],))
    x = torch.stack([data[i:i + config['block_size']] for i in ix])
    y = torch.stack([data[i+1:i + config['block_size']+1] for i in ix])
    return x,y




