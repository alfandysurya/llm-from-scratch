from dataclasses import dataclass

from torch import nn
from torch.nn import functional as F

import torch
import math


@dataclass
class GPTConfig:
    context_window: int = 1024 #the maximum number of consecutive tokens model can process
    vocab_size: int = 50257 #the number of distinct subword
    n_layer: int = 12 #the number of transformer block
    n_attn_head: int = 12 #the number of attention heads
    n_embd: int = 768 #the number representation of a tokens

class MaskedSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_attn_head == 0 # Ensures the embedding dimension can be split into equal-sized feature vectors across all parallel attention heads without remainder.
        
        self.n_attn_head = config.n_attn_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("bias", 
                             torch.tril(torch.ones(config.context_window, config.context_window))
                             .view(1, 1, config.context_window, config.context_window)
                             )

    def forward(self, x):
        B, T, C = x.size() # B: batch size, T: sequence length, C: embedding dimension
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # split the output of c_attn into query, key, and value tensors along the last dimension
        q = q.view(B, T, self.n_attn_head, C // self.n_attn_head).transpose(1, 2) # reshape and permute the query tensor to prepare for multi-head attention
        k = k.view(B, T, self.n_attn_head, C // self.n_attn_head).transpose(1, 2) # reshape and permute the key tensor to prepare for multi-head attention
        v = v.view(B, T, self.n_attn_head, C // self.n_attn_head).transpose(1, 2) # reshape and permute the value tensor to prepare for multi-head attention

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf")) # apply the causal mask to prevent attending to future tokens
        att = F.softmax(att, dim=-1) # apply softmax to obtain attention weights

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs))
        y = y.transpose(1, 2).contiguous().view(B, T, C) # reshape and permute the output tensor to match the original input shape
        y = self.c_proj(y)

        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # fully connected conlayer
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_norm_1 = nn.LayerNorm(config.n_embd)
        self.layer_norm_2 = nn.LayerNorm(config.n_embd)
        self.attention = MaskedSelfAttention(config) # this type of attention is where tokens can't see the future tokens. 
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.mlp(self.layer_norm_2(x))
        return x

class GPT(nn.module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd), # weights of token embedding ()
            "wpe": nn.Embedding(config.context_window, config.n_embd), # weights of positional embedding
            "hidden_layer": nn.ModuleList(Block(config) for i in range(config.n_layer)), # hidden layer
            "layer_norm": nn.LayerNorm(config.n_embd) #Layer normalization (GPT2 use pre-LN instead of post-LN)
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)




