from dataclasses import dataclass, asdict
from typing import Optional, List
import yaml


@dataclass
class Config:
    model_name: str = ""
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
    gelu_approximate: str = "tanh"
    intermediate_size: int = 4

    _supported_models = []

    @classmethod
    def from_model(cls, name: str, **kwargs):
        if name not in name_to_config:
            raise ValueError(f"Unknown model type: {name}")
        conf_dict = name_to_config[name]

        conf_dict = conf_dict.copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @classmethod
    def from_file(cls, path, **kwargs):
        with open(path, encoding="utf-8") as fp:
            file_kwargs = yaml.safe_load(fp)
            if file_kwargs is None:
                raise ValueError(f"{path} is empty which is likely unexpected.")
        file_kwargs.update(kwargs)
        return cls(**file_kwargs)

    @property
    def supported_models(self):
        return self._supported_models


gpt2 = [
    # https://huggingface.co/openai-community/gpt2/blob/main/config.json
    dict(
        model_name="gpt2",
        model_type="gpt2",
        block_size=1024,
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
        model_name="gpt2-medium",
        model_type="gpt2",
        block_size=1024,
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
        model_name='gpt2-large',
        model_type="gpt2",
        block_size=1024,
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
        model_name="gpt2-xl",
        model_type='gpt2',
        block_size=1024,
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


Config._supported_models.extend(gpt2)

name_to_config = {config["model_name"]: config for config in Config._supported_models}

if __name__ == "__main__":
    config = Config.from_model("gpt2-xl")
    print(asdict(config))
