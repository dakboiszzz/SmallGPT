import torch
import yaml
from model import SmallGPT
from data import DataPrep

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load data to get vocab
with open(config['data_path'], 'r') as f:
    words = f.read()

data_prep = DataPrep(words)

# Build model
model = SmallGPT(
    vocab_size=data_prep.vocab_size,
    n_emb=config['n_emb'],
    block_size=config['block_size'],
    num_heads=config['num_heads'],
    n_layer=config['n_layer'],
    dropout=0.0  # No dropout for generation
)

# Load checkpoint
checkpoint_path = config['checkpoint_path'].replace('.pt', '_best.pt')  # Load best model
print(f"Loading model from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model from step {checkpoint['step']}")
print(f"Train loss: {checkpoint['train_loss']:.4f}, Val loss: {checkpoint['val_loss']:.4f}")

# Generate
context = torch.zeros((1, 1), dtype=torch.long)
with torch.no_grad():
    text = model.generate(context, max_new_tokens=1000)[0].tolist()

text_decoded = ''.join([data_prep.itos[c] for c in text])
print("\nGenerated text:")
print("="*50)
print(text_decoded)