from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Modality:
    enabled: bool
    latent: (
        torch.Tensor
    )  # Shape: (B, T, D) where B is the batch size, T is the number of tokens, and D is input dimension
    timesteps: torch.Tensor  # Shape: (B, T) where T is the number of timesteps
    positions: (
        torch.Tensor
    )  # Shape: (B, 3, T) for video, where 3 is the number of dimensions and T is the number of tokens
    context: torch.Tensor
    context_mask: torch.Tensor | None
    cross_attention_mask: torch.Tensor | None = None
