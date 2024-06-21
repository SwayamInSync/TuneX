import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from tunex.config import Config


class CasualAttention(nn.Module):
    def __init__(self, config: Config):
        super(CasualAttention, self).__init__()
        assert config.n_embd % config.n_head == 0

        self.config = config
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.proj_dropout = nn.Dropout(config.resid_dropout)

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # not bias, but mask (following the HF naming)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).
                             view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = torch.split(qkv, self.config.n_embd, dim=-1)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, num_head, T, q_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, num_head, T, k_dim)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, num_head, T, v_dim)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, num_head, T, v_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_dropout(self.c_proj(y))
        return y