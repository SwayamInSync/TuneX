import torch
import torch.nn as nn
from tunex.config import Config


class MLP(nn.Module):
    def __init__(self, config: Config):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.n_embd, config.intermediate_size * config.n_embd)
        self.gelu = nn.GELU(approximate=config.gelu_approximate)
        self.c_proj = nn.Linear(config.intermediate_size * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
