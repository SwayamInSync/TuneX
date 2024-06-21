import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from huggingface_hub import snapshot_download
from tunex.utils.convert_from_hf_weights import convert_and_save_hf_checkpoint
from tunex.utils.utilities import display_title


def find_weight_files(repo_id, access_token):
    from huggingface_hub import repo_info
    from huggingface_hub.utils import filter_repo_objects

    info = repo_info(repo_id, token=access_token)
    filenames = [f.rfilename for f in info.siblings]
    bins = list(filter_repo_objects(items=filenames, allow_patterns=["*.bin*"]))
    safetensors = list(filter_repo_objects(items=filenames, allow_patterns=["*.safetensors"]))
    return bins, safetensors


def download_from_hub(
        repo_id: str,
        access_token: Optional[str] = os.getenv("HF_TOKEN"),
        tokenizer_only: bool = False,
        convert_checkpoint: bool = True,
        checkpoint_dir: Path = Path("checkpoints"),
        model_name: Optional[str] = None,
) -> None:
    """Download weights or tokenizer data from the Hugging Face Hub.

    Arguments:
        repo_id: The repository ID in the format ``org/name`` or ``user/name`` as shown in Hugging Face.
        access_token: Hugging Face API token to access models with restrictions.
        tokenizer_only: Whether to download only the tokenizer files.
        convert_checkpoint: Whether to convert the checkpoint files from hugging face format after downloading.
        checkpoint_dir: Where to save the downloaded files.
        model_name: The existing config name to use for this repo_id. This is useful to download alternative weights of
            existing architectures.
    """
    display_title()

    if model_name is None:
        model_name = repo_id.split("/")[-1].strip()
    files_to_download = ["tokenizer*", "generation_config.json", "config.json"]
    from_safetensors = False
    if not tokenizer_only:
        bins, safetensors = find_weight_files(repo_id, access_token)
        if bins:
            files_to_download.append("*.bin*")  # additional  .bin files
        elif safetensors:
            files_to_download.append("*.safetensors")
            from_safetensors = True
        else:
            raise ValueError(f"Couldn't find weight files for {repo_id}")

    directory = checkpoint_dir / repo_id
    snapshot_download(repo_id, local_dir=directory,
                      allow_patterns=files_to_download,
                      token=access_token,
                      )

    if from_safetensors:
        from safetensors import SafetensorError
        from safetensors.torch import load_file as safetensors_load

        print("Converting .safetensor files to PyTorch binaries (.bin)")
        for safetensor_path in directory.glob("*.safetensors"):
            bin_path = safetensor_path.with_suffix(".bin")
            try:
                result = safetensors_load(safetensor_path)
            except SafetensorError as e:
                raise RuntimeError(f"{safetensor_path} is likely corrupted. Please try to re-download it.") from e
            print(f"{safetensor_path} --> {bin_path}")
            torch.save(result, bin_path)
            os.remove(safetensor_path)

    if convert_checkpoint and not tokenizer_only:
        print("Converting checkpoint files to TuneX format.")
        convert_and_save_hf_checkpoint(checkpoint_dir=directory, model_name=model_name)
    print(f"Weights are converted and saved at {directory}")