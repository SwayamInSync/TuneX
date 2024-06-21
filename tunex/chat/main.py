import time
from pathlib import Path
from typing import Iterator, List, Tuple
import torch

from tunex.config import Config
from tunex.text_generation.main import stochastic_sampling
from tunex.tokenizer import Tokenizer
from tunex.utils.utilities import load_model_from_config, seed_everything, display_title

SEED = 42
seed_everything(SEED)


@torch.inference_mode()
def generate(model, prompt: torch.Tensor, top_p: float, top_k: int, temperature: float,
             max_returned_tokens: int, stop_tokens: Tuple[List[int], ...] = ()) -> Iterator[torch.Tensor]:
    if prompt.ndim == 1:
        prompt = prompt.unsqueeze(0)

    T = prompt.size(0)
    buffer_length = max((len(tokens) for tokens in stop_tokens), default=1)
    yield_i = 0
    tokens = []
    x = prompt
    for i in range(1, max_returned_tokens - T + 1):
        logits = model(x)
        new_token = stochastic_sampling(logits, top_k=top_k, top_p=top_p, temperature=temperature)
        tokens.append(new_token)
        x = torch.cat((x, new_token.unsqueeze(0)), dim=-1)

        # checking the stop tokens match
        for st in stop_tokens:
            l = len(st)
            if l <= len(tokens):
                if all(a == b for a, b in zip(tokens[-l:], st)):
                    return

        if i - yield_i >= buffer_length:
            yield from tokens[yield_i:i]
            yield_i = i


def decode(tokenizer, token_stream):
    tokens_generated = 0
    try:
        for token in token_stream:
            print(tokenizer.decode(token), end="", flush=True)
            tokens_generated += 1
    except KeyboardInterrupt:
        return tokens_generated
    return tokens_generated


def initialize_interface(model, tokenizer: Tokenizer, temperature, top_k, top_p, stop_tokens, device):
    display_title()
    print(f"Initiating chat mode with {model.config.model_name}\n")
    print(f"Setting sead to {SEED}\n")

    while True:
        try:
            prompt = input(">> Prompt: ")
            # print(">> Prompt: (Type '!exit' on a new line to end input).")
            # prompt_lines = []
            # while True:
            #     line = input()
            #     if line.strip().lower() == "!exit":
            #         break
            #     prompt_lines.append(line)
            # prompt = "\n".join(prompt_lines)
        except KeyboardInterrupt:
            break
        prompt = prompt.lower().strip()
        if not prompt or prompt in ("!quit", "!exit"):
            break
        encoded_inputs = tokenizer.encode(prompt, device=device)
        token_stream = generate(model, encoded_inputs, top_p, top_k,
                                temperature, model.max_sequence_length, stop_tokens)
        print(">> Response: ", end="")

        t0 = time.perf_counter()
        tokens_generated = decode(tokenizer, token_stream)
        t = time.perf_counter() - t0

        print(
            f"\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec,"
            f" {tokens_generated} tokens",
        )


@torch.inference_mode()
def main(checkpoint_dir: str,
         top_k: int = 200,
         top_p: float = 1.0,
         temperature: float = 0.99,
         device: torch.device = torch.device("cpu")
         ) -> None:
    """Chat with the model.

        Arguments:
            checkpoint_dir: Path to downloaded model directory.
            top_k: The number of top most probable tokens to consider in the sampling process.
            top_p: Represents the cumulative probability threshold to consider in the sampling process.
            temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
            device: The device to run the model on (cpu, cuda:0, cuda:1, etc.).
        """
    checkpoint_dir = Path(checkpoint_dir)
    # todo: check if checkpoint dir exist else download it from the hub
    checkpoint_path = checkpoint_dir / "tunex_model.pth"

    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    model = load_model_from_config(config)

    model.load_state_dict(torch.load(checkpoint_path), strict=True)
    tokenizer = Tokenizer(checkpoint_dir)

    stop_tokens = ([tokenizer.eos_id],)
    model.eval()
    initialize_interface(model, tokenizer, temperature, top_k, top_p, stop_tokens, device)
