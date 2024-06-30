from __future__ import annotations

import math
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import torch
import torch.nn as nn

from .base import KanLayerBase


def sample_domain(
    domain: tuple[float, float],
    n: int,
    method: Literal["linear", "chebyshev"]
) -> torch.Tensor:
    if method not in ["linear", "chebyshev"]:
        raise ValueError(f"Unsupported method: {method}")
    if method == "linear":
        return torch.linspace(domain[0], domain[1], n)
    if method == "chebyshev":
        xs:torch.Tensor = (torch.cos(torch.arange(n) * math.pi / (n - 1))).sort().values
        return 0.5 * (xs + 1.0) * (domain[1] - domain[0]) + domain[0]
    
def get_fixed_barycentric_weights(method: Literal["linear", "chebyshev"], num_nodes:int) -> torch.Tensor:
    if method == "linear":
        j = torch.arange(num_nodes)
        return ((-1.0)**j) * torch.as_tensor(scipy.special.binom(num_nodes - 1, j))
    if method == "chebyshev":
        wj = torch.ones(num_nodes)
        wj[1::2] = -1
        wj[[0, -1]] *= 0.5
        return wj
    
class LagrangeKan(KanLayerBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        nodes: int = 3,
        domain: tuple[float, float] = (-1, 1),
        node_positions:Literal['linear', 'chebyshev'] | None = None
    ) -> None:
        super().__init__(
            in_features=in_features, 
            out_features=out_features
        )
        self.num_nodes = nodes

        node_positions = node_positions or 'chebyshev'
        
        self.nodes = nn.Parameter(
            sample_domain(domain, self.num_nodes, node_positions).view(1, 1, -1).repeat(out_features, in_features, 1)
        )

        sqrt_k = math.sqrt(1.0 / in_features)
        _control_points = self.nodes.data.clone()
        _control_points = _control_points * torch.empty((out_features, in_features, 1)).uniform_(-sqrt_k, sqrt_k)
        _control_points = _control_points + torch.empty_like(_control_points).normal_(0.0, 0.01)
        self.control_points = nn.Parameter(_control_points)


    def get_pruned(self, in_features: list[int], out_features: list[int]) -> LagrangeKan:
        layer = LagrangeKan(
            in_features=len(in_features),
            out_features=len(out_features),
            nodes=self.num_nodes
        ).to(self.control_points.device)

        layer.nodes.data = self.nodes[out_features][:, in_features]
        layer.control_points.data = self.control_points[out_features][:, in_features]
        return layer

    def compute_activations(self, t:torch.Tensor) -> torch.Tensor:
        wj = (self.nodes.unsqueeze(-1) - self.nodes.unsqueeze(-2)) + torch.eye(self.num_nodes)
        wj = 1.0 / wj.prod(-1)
        deltas = (t[..., None, :, None] - self.nodes)
        deltas = torch.where(
            deltas.abs() < np.spacing(1.0), torch.copysign(torch.full_like(deltas, np.spacing(1.0)), deltas), deltas
        )
        return (wj / deltas * self.control_points).sum(dim=-1) / (wj / deltas).sum(dim=-1)
    
    @torch.inference_mode()
    def plot_activation(
        self, 
        activation_id: tuple[int, int], 
        ax: plt.Axes | None = None,
        trange: tuple[int, int] | None = None, 
        n: int = 128, 
        **kwargs
    ) -> None:
        ax = ax or plt.gca()
        color = kwargs.get("color", None)

        nodes = self.nodes[activation_id[0], activation_id[1]]
        if trange is None:
            tmin, tmax = nodes.min().item(), nodes.max().item()
            trange = tmin - 0.2 * (tmax - tmin), tmax + 0.2 * (tmax - tmin)

        super().plot_activation(
            activation_id=activation_id, 
            ax=ax,
            trange=trange, 
            n=n, 
            **kwargs
        )
        ax.scatter(
            nodes.cpu(),
            self.control_points[activation_id[0], activation_id[1]].cpu(),
            color=color
        )
    
class FixedNodesLagrangeKan(KanLayerBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        nodes: int = 3,
        domain: tuple[float, float] = (-1, 1),
        node_positions:Literal['linear', 'chebyshev'] | None = None
    ) -> None:
        """
        Initialize the fixed nodes Lagrange kan layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            nodes (int, optional): Number of nodes. Defaults to 3.
            domain (tuple[float, float], optional): Domain range. Defaults to (-1, 1).
            node_positions (Literal['linear', 'chebyshev'] | None, optional): Method to sample node positions. Defaults to None.
        """
        super().__init__(
            in_features=in_features, 
            out_features=out_features
        )

        self.num_nodes = nodes
        node_positions = node_positions or 'chebyshev'

        xj = sample_domain(domain, self.num_nodes, node_positions).view(1, 1, -1)

        sqrt_k = math.sqrt(1.0 / in_features)

        _control_points = xj.view(1, 1, -1).repeat(out_features, in_features, 1)
        _control_points = _control_points * torch.empty((out_features, in_features, 1)).uniform_(-sqrt_k, sqrt_k)
        _control_points = _control_points + torch.empty_like(_control_points).normal_(0.0, 0.01)

        self.control_points = nn.Parameter(_control_points)
        self.register_buffer("nodes", xj.view(1, 1, -1))
        self.register_buffer(
            "_wj", 
            get_fixed_barycentric_weights(node_positions, self.num_nodes)
        ) 

    def compute_activations(self, t:torch.Tensor) -> torch.Tensor:
        deltas = (t[..., None, :, None] - self.nodes)
        deltas = torch.where(
            deltas.abs() < 1e-6, torch.copysign(torch.full_like(deltas, 1e-6), deltas), deltas
        )
        return (self._wj / deltas * self.control_points).sum(dim=-1) / (self._wj / deltas).sum(dim=-1)

    def get_pruned(self, in_features: list[int], out_features: list[int]) -> KanLayerBase:
        layer = FixedNodesLagrangeKan(
            in_features=len(in_features),
            out_features=len(out_features),
            nodes=self.num_nodes
        ).to(self.control_points.device)

        layer.control_points.data = self.control_points[out_features][:, in_features]
        layer.nodes = self.nodes
        layer._wj = self._wj
        return layer
    
    @torch.inference_mode()
    def plot_activation(
        self, 
        activation_id: tuple[int, int], 
        ax: plt.Axes | None = None,
        trange: tuple[int, int] | None = None, 
        n: int = 128, 
        **kwargs
    ) -> None:
        ax = ax or plt.gca()
        color = kwargs.get("color", None)

        nodes = self.nodes[0, 0]
        if trange is None:
            tmin, tmax = nodes.min().item(), nodes.max().item()
            trange = tmin - 0.2 * (tmax - tmin), tmax + 0.2 * (tmax - tmin)

        super().plot_activation(
            activation_id=activation_id, 
            ax=ax,
            trange=trange, 
            n=n, 
            **kwargs
        )
        ax.scatter(
            nodes.cpu(),
            self.control_points[activation_id[0], activation_id[1]].cpu(),
            color=color
        )

