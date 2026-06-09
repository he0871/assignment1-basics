import numpy.typing as npt
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor
import cs336_basics.models.basic as basic
from collections.abc import Iterable

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"dataset length: {len(dataset)}, batch size: {batch_size}, context length: {context_length}")
    batch_x = []
    batch_labels = []
    for _ in range(batch_size):
        indices = np.random.choice(len(dataset)-1, size=context_length, replace=True)
        labels = indices + 1
        x = dataset[indices]
        labels = dataset[labels]
        batch_x.append(x)
        batch_labels.append(labels)
    batch_x = torch.tensor(batch_x, device=device)
    batch_labels = torch.tensor(batch_labels, device=device)
    return batch_x, batch_labels

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:

    labels = inputs[
        torch.arange(inputs.shape[0]),
        targets
    ]
    max_items = torch.max(inputs, dim=-1, keepdim=True).values
    shifted_inputs = inputs - max_items
    front = max_items + torch.log(torch.sum(torch.exp(shifted_inputs), dim=-1, keepdim=True))
    losses = front - labels
    return torch.mean(losses)

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    total_norm = torch.sqrt(
        sum(torch.sum(p.grad ** 2) for p in parameters if p.grad is not None)
    )
    print(f"total_norm: {total_norm}")
    if total_norm > max_l2_norm:
        for params in parameters:
            if params.grad is None:
                continue
            params.grad *= (max_l2_norm / (total_norm + eps))