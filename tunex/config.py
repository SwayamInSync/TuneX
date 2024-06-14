from dataclasses import dataclass, asdict


@dataclass
class Config:
    model_type: str = ""
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    embd_dropout: float = 0.0
    bias: bool = True
    lm_head_bias: bool = False
    gelu_approximate: str = ""

    @classmethod
    def from_model(cls, name: str, **kwargs):
        if name not in name_to_config:
            raise ValueError(f"Unknown model type: {name}")
        conf_dict = name_to_config[name]

        conf_dict = conf_dict.copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)


configs = []
gpt2 = [
    # https://huggingface.co/openai-community/gpt2/blob/main/config.json
    dict(
        model_type="gpt2",
        n_layer=12,
        n_head=12,
        n_embd=768,
        vocab_size=50257,
        attn_dropout=0.1,
        resid_dropout=0.1,
        embd_dropout=0.1,
        gelu_approximate="tanh",
    ),
    # https://huggingface.co/openai-community/gpt2-medium/blob/main/config.json
    dict(
        model_type="gpt2-medium",
        n_layer=24,
        n_head=16,
        n_embd=1024,
        vocab_size=50257,
        attn_dropout=0.1,
        resid_dropout=0.1,
        embd_dropout=0.1,
        gelu_approximate="tanh",
    ),
    # https://huggingface.co/openai-community/gpt2-large/blob/main/config.json
    dict(
        model_type='gpt2-large',
        n_layer=36,
        n_head=20,
        n_embd=1280,
        vocab_size=50257,
        attn_dropout=0.1,
        resid_dropout=0.1,
        embd_dropout=0.1,
        gelu_approximate="tanh",
    ),
    # https://huggingface.co/openai-community/gpt2-xl/blob/main/config.json
    dict(
        model_type='gpt2-xl',
        n_layer=48,
        n_head=25,
        n_embd=1600,
        vocab_size=50257,
        attn_dropout=0.1,
        resid_dropout=0.1,
        embd_dropout=0.1,
        gelu_approximate="tanh",
    )
]
configs.extend(gpt2)

name_to_config = {config["model_type"]: config for config in configs}

if __name__ == "__main__":
    config = Config.from_model("gpt2")
    print(asdict(config))
