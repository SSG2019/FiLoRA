import re
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import List, Dict, Tuple, Pattern, Union, Optional, Sequence

def compute_UV_LoRA_XS(
        model: nn.Module,
        layer_names: List[str],
        rank: int
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:

    results = {}

    for layer_name in layer_names:
        module = model
        for part in layer_name.split('.'):
            module = getattr(module, part)

        if not isinstance(module, nn.Linear):
            print(f"Warning: {layer_name} is not a linear layer, skipping.")
            continue

        W = module.weight.data

        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_r = U[:, :rank].contiguous()
        V_r = Vh[:rank, :].T.contiguous()

        S_r = torch.diag(S[:rank])

        U_r = torch.matmul(U_r, S_r)

        results[layer_name] = (U_r, V_r)
    return results

def compute_UV_random(
        model: nn.Module,
        layer_names: List[str],
        rank: int
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:

    results = {}

    for layer_name in layer_names:

        module = model
        for part in layer_name.split('.'):
            module = getattr(module, part)

        if not isinstance(module, nn.Linear):
            print(f"Warning: {layer_name} is not a linear layer, skipping.")
            continue

        out_dim, in_dim = module.weight.shape

        A = torch.randn(out_dim, out_dim, device=module.weight.device)
        B = torch.randn(in_dim, in_dim, device=module.weight.device)

        A = (A + A.T) / 2
        B = (B + B.T) / 2

        eigvals_A, eigvecs_A = torch.linalg.eigh(A)
        eigvals_B, eigvecs_B = torch.linalg.eigh(B)

        idx_A = torch.argsort(eigvals_A, descending=True)[:rank]
        idx_B = torch.argsort(eigvals_B, descending=True)[:rank]

        U_r = eigvecs_A[:, idx_A].contiguous()
        V_r = eigvecs_B[:, idx_B].contiguous()

        results[layer_name] = (U_r, V_r)

    return results


class VeRALinear(nn.Module):
    """
    VeRA (non-shared) version — each layer has its own random A, B matrices.
    Reference: VeRA: Vector-based Random Matrix Adaptation (ICLR 2024)
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 256, d_init: float = 0.1):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank

        target_device = base_linear.weight.device

        # Store frozen pretrained weight (already on target_device if base_linear was moved)
        self.W0 = base_linear.weight.data.clone().detach()
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.data.clone().detach())
        else:
            self.bias = None

        # Independent frozen random matrices per layer, created directly on the target_device
        # This is the key change to prevent the RuntimeError
        self.register_buffer("A", torch.randn(rank, self.in_features, device=target_device) * 0.02)
        self.register_buffer("B", torch.randn(self.out_features, rank, device=target_device) * 0.02)

        # Trainable scaling vectors (also created on the target_device)
        self.d = nn.Parameter(torch.ones(rank, device=target_device) * d_init)
        self.b = nn.Parameter(torch.zeros(self.out_features, device=target_device))

    def forward(self, x):
        # Base projection
        out = F.linear(x, self.W0, self.bias)

        # Adaptation term
        # At this point, x, self.A, self.d, self.B, self.b should all be on the same device
        Ax = F.linear(x, self.A)            # (batch, rank)
        Ax = Ax * self.d                    # apply Λ_d
        delta = F.linear(Ax, self.B)        # (batch, out_features)
        delta = delta * self.b              # apply Λ_b

        return out + delta

def replace_one_with_vera(model: nn.Module, layer_name: str, rank: int = 256, d_init: float = 0.1):
    """
    Replace a target nn.Linear layer with a VeRALinear (non-shared version).
    """
    parts = layer_name.split(".")
    submodule = model
    for p in parts[:-1]:
        submodule = getattr(submodule, p)

    target = getattr(submodule, parts[-1])
    assert isinstance(target, nn.Linear), f"{layer_name} is not nn.Linear"

    new_layer = VeRALinear(target, rank=rank, d_init=d_init)
    setattr(submodule, parts[-1], new_layer)

    return model

def replace_layers_with_vera(model: nn.Module, layer_names: list, rank: int = 256, d_init: float = 0.1):
    for name in layer_names:
        # Use a try-except block to gracefully handle cases where a specified layer_name might not exist
        try:
            model = replace_one_with_vera(model, name, rank=rank, d_init=d_init)
        except AttributeError as e:
            print(f"Warning: Could not replace layer '{name}'. Error: {e}")
        except AssertionError as e:
            print(f"Warning: Could not replace layer '{name}'. Error: {e}")
    return model