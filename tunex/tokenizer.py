import json
import torch
from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.bos_id = None
        self.eos_id = None
        if not (self.checkpoint_dir / "tokenizer.json").is_file():
            raise FileNotFoundError(f"Tokenizer.json not found in the {checkpoint_dir}")

        self.processor = HFTokenizer.from_file((self.checkpoint_dir / "tokenizer.json").as_posix())

        if (special_tokens_path := self.checkpoint_dir / "tokenizer_config.json").is_file():
            with open(special_tokens_path, encoding="utf-8") as fp:
                config = json.load(fp)
            bos_token = config.get("bos_token")
            self.bos_id = self.token_to_id(bos_token) if bos_token is not None else None
            eos_token = config.get("eos_token")
            self.eos_id = self.token_to_id(eos_token) if eos_token is not None else None
        if (special_tokens_path := self.checkpoint_dir / "generation_config.json").is_file():
            with open(special_tokens_path, encoding="utf-8") as fp:
                config = json.load(fp)
            if self.bos_id is None:
                self.bos_id = config.get("bos_token_id")
            if self.eos_id is None:
                self.eos_id = config.get("eos_token_id")

    @property
    def vocab_size(self) -> int:
        return self.processor.get_vocab_size(with_added_tokens=False)

    def token_to_id(self, token: str) -> int:
        id_ = self.processor.token_to_id(token)
        if id_ is None:
            raise ValueError(f"token {token!r} not found in the collection.")
        return id_

    def encode(self, string, device, max_length=-1) -> torch.Tensor:
        tokens = self.processor.encode(string).ids
        if self.bos_id:
            tokens = [self.bos_id] + tokens
        if self.eos_id:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tokens) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = [tokens.item()] if tokens.ndim == 0 else tokens.tolist()
        decoded_string = self.processor.decode(tokens)
        return decoded_string

