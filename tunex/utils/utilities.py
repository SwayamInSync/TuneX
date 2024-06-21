import yaml
import pyfiglet
from pathlib import Path
from tunex import Config
from tunex.models import *
from dataclasses import asdict

import random
import numpy as np
import torch


def seed_everything(seed: int) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def load_model_from_config(config: Config):
    if config.model_type == "gpt2":
        model = GPT2(config=config)
        return model


def save_config(config: Config, checkpoint_dir: Path) -> None:
    config_dict = asdict(config)
    with open(checkpoint_dir / "model_config.yaml", "w", encoding="utf-8") as fp:
        yaml.dump(config_dict, fp)


def display_title():
    print(pyfiglet.figlet_format("TuneX"))


def list_supported_models() -> None:
    """List the models supported by TuneX

        Arguments:

        """
    display_title()
    print("Supported models: \n")
    model_names = [c['model_name'] for c in Config._supported_models]
    for i, model_name in enumerate(model_names):
        print(i+1, model_name)
