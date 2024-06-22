import torch
from torch import nn

from tunex.config import Config
from tunex.modules.attention import CasualAttention
from tunex.modules.mlp import MLP


class Block(nn.Module):
    def __init__(self, config: Config):
        super(Block, self).__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: Config):
        super(GPT2, self).__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe=nn.Embedding(self.config.block_size, self.config.n_embd),
            drop=nn.Dropout(self.config.embd_dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(self.config.n_embd),
        ))

        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=config.lm_head_bias)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        self.max_sequence_length = config.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, 0, 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, 0, 0.02)

    def forward(self, x):
        B, T = x.size()
        assert T <= self.config.block_size, f"{T} should be <= {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(x)
        x = tok_emb + pos_emb
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)
        return x
