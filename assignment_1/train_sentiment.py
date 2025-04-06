import os
import time
import math
import pickle
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sklearn.metrics as metrics
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from contextlib import nullcontext
from model_sentiment import GPTConfig, GPTforClassification, GPT2Wrapper
import torch._dynamo # added to suppress triton exception

# Suppress Torch Dynamo errors to prevent runtime crashes during compilation
torch._dynamo.config.suppress_errors = True

# --- Configuration ---
out_dir = "out"
data_dir = "data/sentiment"
use_gpt2 = False  # Set to False to use the NanoGPT-style model instead

# Model & optimization hyperparameters
batch_size = 16
block_size = 256
max_iters = 10000
eval_interval = 200
learning_rate = 5e-5 if use_gpt2 else 3e-4
weight_decay = 0.05
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.1
bias = True
device = "cuda"
dtype = "float16"
compile_model = True
wandb_project = "sentiment_gpt"
wandb_run_name = f"gpt2_finetune" if use_gpt2 else "nanogpt_train"

# --- Distributed Training Setup ---
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
else:
    master_process = True

# Ensure output directory exists
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ptdtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)

# --- Data Loading Helpers ---
def load_pickle(name):
    with open(os.path.join(data_dir, name), 'rb') as f:
        return pickle.load(f)

data_suffix = "_gpt2.pkl" if use_gpt2 else ".pkl"
train_data = load_pickle(f"train_data{data_suffix}")
val_data = load_pickle(f"val_data{data_suffix}")
test_data = load_pickle(f"test_data{data_suffix}")
meta = load_pickle("meta_gpt2.pkl" if use_gpt2 else "meta.pkl")

# --- Dataset and Dataloader ---
class ClassificationDataset(Dataset):
    def __init__(self, encoded_conversations, labels):
        self.X = encoded_conversations
        self.Y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx][:block_size]  # Truncate to block size
        return torch.tensor(x, dtype=torch.long), torch.tensor(self.Y[idx], dtype=torch.long)

def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    x_padded = [torch.cat([x, torch.full((max_len - len(x),), 0, dtype=torch.long)]) for x in xs]
    attn_masks = [torch.cat([torch.ones(len(x)), torch.zeros(max_len - len(x))]) for x in xs]
    return torch.stack(x_padded), torch.stack(attn_masks), torch.tensor(ys)

# Load training, validation, and test data
train_loader = DataLoader(ClassificationDataset(train_data['encoded_conversations'], train_data['labels']), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(ClassificationDataset(val_data['encoded_conversations'], val_data['labels']),     batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(ClassificationDataset(test_data['encoded_conversations'], test_data['labels']),   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# --- Model Initialization ---
if use_gpt2:
    model = GPT2Wrapper(num_classes=meta['num_classes']).to(device)
else:
    gpt_config = GPTConfig(
        block_size=block_size,
        vocab_size=meta['vocab_size'],
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias
    )
    model = GPTforClassification(gpt_config, num_classes=meta['num_classes']).to(device)

# Optimizer setup
if hasattr(model, 'configure_optimizers'):
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2))
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

# Compile model for performance (optional, PyTorch 2.0+)
if compile_model:
    model = torch.compile(model)

# Wrap model in DistributedDataParallel if using DDP
if ddp:
    model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])

# Optional: Freeze GPT-2 layers to avoid overfitting on small datasets
if use_gpt2:
    for param in model.gpt2.parameters():
        param.requires_grad = False

# Gradient scaler for mixed-precision training
scaler = torch.cuda.amp.GradScaler(enabled=(dtype=='float16'))

# --- Weights & Biases Init ---
if master_process:
    wandb.init(project=wandb_project, name=wandb_run_name, config={
        "use_gpt2": use_gpt2,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "block_size": block_size,
        "max_iters": max_iters
    })

# --- Evaluation Logic ---
def evaluate_split(dloader):
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for X, attn, Y in dloader:
            X, attn, Y = X.to(device), attn.to(device), Y.to(device)
            with ctx:
                if use_gpt2:
                    logits, loss = model(X, attn, Y)
                else:
                    logits, loss = model(X, Y)
            losses.append(loss.item())
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(Y.cpu().tolist())
    return np.mean(losses), preds, labels

# --- Training Loop ---
best_val_loss = 1e9
patience_counter = 0
early_stop_patience = 5
train_iter = iter(train_loader)

for step in range(max_iters):
    model.train()
    try:
        X, attn, Y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        X, attn, Y = next(train_iter)

    X, attn, Y = X.to(device), attn.to(device), Y.to(device)
    with ctx:
        if use_gpt2:
            logits, loss = model(X, attn, Y)
        else:
            logits, loss = model(X, Y)

    # Standard backward pass and optimization
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()

    # Periodic evaluation and early stopping
    if step % eval_interval == 0:
        train_loss, train_preds, train_labels = evaluate_split(train_loader)
        val_loss, val_preds, val_labels = evaluate_split(val_loader)
        print(f"step {step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        if master_process:
            train_acc = metrics.accuracy_score(train_labels, train_preds)
            val_acc = metrics.accuracy_score(val_labels, val_preds)
            wandb.log({
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc
            })
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model": model.state_dict()}, os.path.join(out_dir, "ckpt.pt"))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > early_stop_patience:
                    print("Stopping early")
                    break

# --- Final Test Evaluation ---
if master_process:
    test_loss, test_preds, test_labels = evaluate_split(test_loader)
    test_acc = metrics.accuracy_score(test_labels, test_preds)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    print(metrics.classification_report(test_labels, test_preds, target_names=meta['labels']))

    # Confusion matrix plot
    cm = metrics.confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=meta['labels'], yticklabels=meta['labels'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Test Confusion Matrix")
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()

    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_acc
    })

# Cleanup DDP if applicable
if ddp:
    destroy_process_group()
