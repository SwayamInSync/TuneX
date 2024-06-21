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

    @classmethod
    def from_pretrained(cls, model_type, debug=False):
        from huggingface_hub import hf_hub_download

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints

        config = Config(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # downloading weights
        # weights_path = hf_hub_download(repo_id=model_type, filename="pytorch_model.bin")
        if not debug:
            sd_hf = torch.load("checkpoints/gpt2/pytorch_model.bin")
        else:
            sd_hf = torch.load("../../../checkpoints/gpt2/pytorch_model.bin")
        sd_hf['lm_head.weight'] = sd_hf['wte.weight']
        sd_keys_hf = [k for k in sd_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            p_k = "transformer." + k if k != "lm_head.weight" else k
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[p_k].shape
                with torch.no_grad():
                    sd[p_k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[p_k].shape
                with torch.no_grad():
                    sd[p_k].copy_(sd_hf[k])

        print(model.state_dict().keys())
        print(sd_hf.keys())

        return model


if __name__ == '__main__':
    # model = GPT.from_pretrained("gpt2")
    # model = GPT2(Config.from_model("gpt2"))
    model = GPT2.from_pretrained("gpt2", debug=True)
    print("worked")
