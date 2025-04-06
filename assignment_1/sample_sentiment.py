"""
Sample sentiment predictions from a trained classification model.
"""

import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model_sentiment import GPTConfig, GPTforClassification  # make sure this class is defined in your model.py

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = 'out'
start = "This product is amazing and works flawlessly!"  # Replace or set to "FILE:prompt.txt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------
model_args = {
    'n_layer': 6,
    'n_head': 6,
    'n_embd': 384,
    'block_size': 256,
    'bias': False,
    'vocab_size': 50304,  # or whatever was used
}
gptconf = GPTConfig(**model_args)
model = GPTforClassification(gptconf, num_classes=3)  # adjust num_classes if needed

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load tokenizer (encoding)
meta_path = os.path.join(out_dir, 'meta.pkl')  # replace with actual path
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi.get(c, stoi[' ']) for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s)
    decode = lambda l: enc.decode(l)

# Load model
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)

model = GPTforClassification(GPTConfig(**model_args))
state_dict = checkpoint['model']

# Remove any unwanted prefixes from keys (for DDP-trained models)
unwanted_prefix = '_orig_mod.'
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()
if compile:
    model = torch.compile(model)

# Encode and prepare input
if start.startswith("FILE:"):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

# Predict
with torch.no_grad():
    with ctx:
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

# Display result
label_map = {0: "negative", 1: "neutral", 2: "positive"}  # adjust as needed
print(f"\nInput: {start}")
print(f"Predicted sentiment: {label_map[pred]}")
print(f"Class probabilities: {probs.cpu().numpy()}")
