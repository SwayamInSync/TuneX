import torch
import torch.nn.functional as F

from tunex.config import Config
from tunex.tokenizer import Tokenizer


def top_p_sample(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    sorted_indices_to_remove[-1:] = 0
    indices_to_remove = torch.scatter(sorted_indices_to_remove, 0, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def top_k_sample(logits: torch.Tensor, topk: int) -> torch.Tensor:
    values, indices = torch.topk(logits, k=min(topk, logits.size(-1)), dim=-1)
    return torch.full_like(logits, float("-inf")).scatter(-1, indices, values)


def multinomial_num_samples(logits: torch.Tensor) -> torch.Tensor:
    # using Gumbel noise trick for fast alternative
    gumbel_noise = -torch.empty_like(logits).exponential_(1).log()
    noisy_logits = logits + gumbel_noise
    return noisy_logits.argmax(dim=-1, keepdim=True)


def stochastic_sampling(logits: torch.Tensor, top_k: int, top_p: float, temperature: float) -> torch.Tensor:
    logits = logits[0, -1]
    if top_k:
        logits = top_k_sample(logits, top_k)
    if temperature > 0.0:
        logits = logits / temperature
    if top_p:
        logits = top_p_sample(logits, top_p)
    return multinomial_num_samples(logits)  # only 1 sample is needed

# todo: Implement beam search sampling


