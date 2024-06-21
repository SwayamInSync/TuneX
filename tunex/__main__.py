import torch
from typing import TYPE_CHECKING, Any, Optional
from jsonargparse import set_config_read_mode, set_docstring_parse_options, CLI

from tunex import download_from_hub
from tunex import main as chat_fn
from tunex.utils.utilities import list_supported_models

if TYPE_CHECKING:
    from jsonargparse import ArgumentParser


def _new_parser(**kwargs: Any) -> "ArgumentParser":
    from jsonargparse import ActionConfigFile, ArgumentParser

    parser = ArgumentParser(**kwargs)
    parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    return parser


def main() -> None:
    parser_data = {
        "download": download_from_hub,
        "chat": chat_fn,
        "list": list_supported_models,
    }

    set_docstring_parse_options(attribute_docstrings=True)
    set_config_read_mode(urls_enabled=True)

    torch.set_float32_matmul_precision("high")
    CLI(parser_data)


if __name__ == "__main__":
    main()
