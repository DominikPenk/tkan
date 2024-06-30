import numpy as np
import torch
import torch.nn as nn

from .activations_tracker import ActivationsTracker
from ..nn.base import KanLinearBase

def kan_node_sparsity_loss(phi:torch.Tensor, eps:float|None=None) -> tuple[torch.Tensor, torch.Tensor]:
    phi = phi.view(-1, *phi.shape[-2:])
    phi_L1 = phi.abs().mean(dim=0)

    eps = eps or np.spacing(1.0)

    L1_norm = phi_L1.sum()
    L1_norm_clamped = L1_norm.clamp_min_(eps)
    entropy = -(phi_L1 / L1_norm_clamped * torch.log((phi_L1 / L1_norm_clamped)).clamp_min_(eps)).sum()

    return L1_norm, entropy

class KanNodeSparsityLoss:
    def __init__(
        self,
        module:nn.Module,
        lambda_norm:float = 1.0,
        lambda_entropy:float = 1.0
    ) -> None:
        self._tracker = ActivationsTracker(module)
        self._tracker.register_hooks()
        self.lambda_norm = lambda_norm
        self.lambda_entropy = lambda_entropy

    def compute(self) -> torch.Tensor:
        loss = 0.0
        for activation in self._tracker.get_activations():
            L1_norm, entropy = kan_node_sparsity_loss(activation)
            loss += L1_norm + entropy
        return loss


    def __del__(self):
        self._tracker.unregister_hooks()

class KanLayerRegularizationLoss:
    def __init__(
        self,
        module:nn.Module,
    ) -> None:
        self._layers:list[KanLinearBase] = [
            layer for layer in module.modules() if isinstance(layer, KanLinearBase)
        ]

    def compute(self) -> torch.Tensor:
        layer_losses = [layer.regularization_loss() for layer in self._layers]
        return sum(layer_losses) / max(len(layer_losses), 1)
