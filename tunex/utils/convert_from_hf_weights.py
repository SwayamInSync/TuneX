import os
import gc
import json
from pathlib import Path
import torch
from typing import Dict, Optional
from functools import partial

from tunex.config import Config
from tunex.utils.utilities import save_config, load_model_from_config


def get_layer_pos(layer_name: str, idx: int):
    split = layer_name.split(".")
    number = int(split[idx])
    split[idx] = "{}"
    from_name = ".".join(split)
    return from_name, number


def gpt2_checkpointing(state_dict: Dict[str, torch.Tensor], hf_weights) -> None:
    weight_map = {
        "wte.weight": "transformer.wte.weight",
        "wpe.weight": "transformer.wpe.weight",
        "ln_f.bias": "transformer.ln_f.bias",
        "ln_f.weight": "transformer.ln_f.weight",
        "h.{}.attn.bias": "transformer.h.{}.attn.bias",
        "h.{}.ln_1.weight": "transformer.h.{}.ln_1.weight",
        "h.{}.ln_1.bias": "transformer.h.{}.ln_1.bias",
        "h.{}.ln_2.weight": "transformer.h.{}.ln_2.weight",
        "h.{}.ln_2.bias": "transformer.h.{}.ln_2.bias",
        "h.{}.attn.c_attn.weight": "transformer.h.{}.attn.c_attn.weight",
        "h.{}.attn.c_attn.bias": "transformer.h.{}.attn.c_attn.bias",
        "h.{}.attn.c_proj.weight": "transformer.h.{}.attn.c_proj.weight",
        "h.{}.attn.c_proj.bias": "transformer.h.{}.attn.c_proj.bias",
        "h.{}.mlp.c_fc.weight": "transformer.h.{}.mlp.c_fc.weight",
        "h.{}.mlp.c_fc.bias": "transformer.h.{}.mlp.c_fc.bias",
        "h.{}.mlp.c_proj.weight": "transformer.h.{}.mlp.c_proj.weight",
        "h.{}.mlp.c_proj.bias": "transformer.h.{}.mlp.c_proj.bias",
    }

    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

    for name, param in hf_weights.items():
        print(name)
        if "h." in name:
            from_name, number = get_layer_pos(name, 1)
            to_name = weight_map[from_name].format(number)
        else:
            to_name = weight_map[name]

        if any(k in name for k in transposed):
            param = param.t()
        state_dict[to_name] = param

    # initializing the lm_head weights
    state_dict["lm_head.weight"] = hf_weights["wte.weight"]


def convert_and_save_hf_checkpoint(checkpoint_dir: Path, model_name: str) -> None:
    config = Config.from_model(model_name)
    save_config(config, checkpoint_dir)

    copy_fn = None
    if config.model_type == "gpt2":
        copy_fn = partial(gpt2_checkpointing)

    if copy_fn is None:
        raise ValueError(f"No conversion function corresponding to {model_name} is found")

    state_dict = {}

    # pytorch_bin_map_json_path = checkpoint_dir / "pytorch_model.bin.index.json"
    # if pytorch_bin_map_json_path.is_file():  # not all checkpoints have this file
    #     with open(pytorch_bin_map_json_path, encoding="utf-8") as json_map:
    #         bin_index = json.load(json_map)
    #     bin_files = {checkpoint_dir / bin_ for bin_ in bin_index["weight_map"].values()}
    # else:
    #     bin_files = set(checkpoint_dir.glob("*.bin"))
    #     bin_files = {f for f in bin_files if f.name != "training_args.bin"}
    # if not bin_files:
    #     raise ValueError(f"Expected {str(checkpoint_dir)!r} to contain .bin files")
    bin_file = [i for i in checkpoint_dir.glob("*.bin")][0]

    print("Processing", bin_file)
    hf_weights = torch.load(bin_file)
    copy_fn(state_dict, hf_weights)
    gc.collect()
    print(f"Saving converted checkpoint to {checkpoint_dir}")

    # Save the model state dictionary
    torch.save(state_dict, (checkpoint_dir / "tunex_model.pth").as_posix())


