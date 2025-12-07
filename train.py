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




