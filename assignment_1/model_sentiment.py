# model_sentiment.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import GPT2Model

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50304
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True

class GPTforClassification(nn.Module):
    """
    A minimal implementation of a transformer-style model for classification.
    This mimics NanoGPT but simplified, built with PyTorch TransformerEncoder.
    """
    def __init__(self, config, num_classes):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.n_embd,
                nhead=config.n_head,
                dim_feedforward=4 * config.n_embd,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True
            ),
            num_layers=config.n_layer
        )
        self.classifier = nn.Linear(config.n_embd, num_classes)

    def forward(self, x, targets=None):
        x = self.embed(x)                      # [B, T] -> [B, T, C]
        x = self.transformer(x)                # Transformer encoding
        x = x.mean(dim=1)                      # Pooling: average over tokens
        logits = self.classifier(x)            # Final logits
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        return logits, None

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        # Defensive check to ensure model has trainable parameters
        param_list = list(self.parameters())
        if len(param_list) == 0:
            raise ValueError("GPTforClassification has no trainable parameters.")
        return torch.optim.AdamW(param_list, lr=learning_rate, betas=betas, weight_decay=weight_decay)

class GPT2Wrapper(nn.Module):
    """
    Fine-tune GPT-2 for classification with 3 classes.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        self.classifier = nn.Linear(self.gpt2.config.hidden_size, num_classes)
        # for checkpoint
        self.config = self.gpt2.config

    def forward(self, input_ids, attention_mask=None, targets=None):
        """
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len], 1 for real tokens, 0 for pad
        targets: [batch_size] with label in {0,1,2}
        """
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        # average hidden states
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        return logits, None

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # typical weight decay grouping
            if len(param.shape) == 1 or name.endswith("bias") or "LayerNorm.weight" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        grouped = [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}
        ]
        return torch.optim.AdamW(grouped, lr=learning_rate, betas=betas)
