import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

class KanLinearBase(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def compute_activations(self, x:torch.Tensor) -> torch.Tensor:
        """Compute activation matrix.
        
        Args:
            x (torch.Tensor): input tensor of shape (*, in_features)

        Returns:
            torch.Tensor: activation matrix of shape (*, out_features, in_features)
        """
        raise NotImplementedError("Must be implemented in subclass")
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Input tensor must have shape (*, {self.in_features}), got {list(x.shape)}")

        activations = self.compute_activations(x)

        return activations.sum(dim=-1)
    
    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device, dtype=torch.float32)

    def get_pruned(self, in_features: list[int], out_features: list[int]) -> 'KanLinearBase':
        """Return a smaller Layer with only the given in and out features.
        
        Args:
            in_features (list[int]): list of in features to keep
            out_features (list[int]): list of out features to keep

        Returns:
            nn.Module: a new layer with only the given in and out features
        """
        raise NotImplementedError(f"Pruning not implemented for {self.__class__.__name__}")

    def init_with_non_linearity(
        self, 
        non_linearity:Callable[[torch.Tensor], torch.Tensor],
        domain:tuple[int, int]=(-1, 1)
    ) -> float:
        raise NotImplementedError("Not implemented")

    @torch.inference_mode()
    def plot_activation(
        self, 
        activation_id:tuple[int, int],
        ax:plt.Axes | None = None,
        n:int = 128,
        trange:tuple[int, int] | None = None,
        **kwargs
    ) -> None:
        """Plot the requested activation to the given axis.

        Args:
            activation_id (tuple[int, int]): tuple (output_feature, input_feature) indicating which activation to plot
            ax (plt.Axes | None, optional): Axes to plot to. If none, use the current axis (plt.gca()). Defaults to None
            n (int, optional): Number of points for plotting. Defaults to 128.
            trange (tuple[int, int] | None, optional): Minimum and maximum input value. Defaults to None.
        """
        ax = ax or plt.gca()
        if trange is None:
            trange = (-1, 1)
        device = next(self.parameters()).device
        t = torch.linspace(*trange, n, device=device).view(-1, 1).repeat_interleave(self.in_features, dim=-1)

        activation = self.compute_activations(t)[:, activation_id[0], activation_id[1]]
        ax.plot(t.cpu(), activation.cpu(), **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.in_features}, {self.out_features}]"

class PolynomialKanLinear(KanLinearBase):
    def __init__(
        self,
        in_features:int,
        out_features:int,
        order:int = 3,
        bias:bool = True,
        normalize:bool=False
    ) -> None:
        super().__init__(
            in_features=in_features, 
            out_features=out_features
        )
        
        self.order = order
        self.normalize = normalize  # TODO: move this to a property to ensure correct write access
        self.bias   = nn.Parameter(torch.zeros(out_features, 1), requires_grad=bias) if bias else None 
        self.scale  = nn.Parameter(torch.ones(in_features), requires_grad=normalize)
        self.offset = nn.Parameter(torch.zeros(in_features), requires_grad=normalize)
        self.control_points = nn.Parameter(self.get_initial_control_points())
        if self.bias is not None:
            nn.init.normal_(self.bias, 0, 1.0 / math.sqrt(out_features))

    def get_initial_control_points(self) -> torch.Tensor:
        """Generate a set of initial control points."""
        return torch.empty(self.out_features, self.in_features, self.order).normal(0, 0.1)

    def evaluate_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the polynomial basis functions at x."""
        raise NotImplementedError("Must be implemented in subclass")
    
    def regularization_loss(self) -> torch.Tensor:
        return self.control_points[..., :].abs().sum()
    
    def get_pruned(self, in_features: list[int], out_features: list[int]) -> 'PolynomialKanLinear':
        cls = self.__class__
        layer = cls(
            in_features=len(in_features),
            out_features=len(out_features),
            order=self.order,
            bias=self.bias is not None,
            normalize=self.normalize
        )
        layer.control_points.data = self.control_points[out_features][:, in_features]
        layer.scale.data = self.scale[in_features]
        layer.offset.data = self.offset[in_features]
        if layer.bias is not None:
            layer.bias.data = self.bias[out_features]
        return layer

    def init_with_non_linearity(
        self, 
        non_linearity: Callable[[torch.Tensor], torch.Tensor], 
        domain: tuple[int, int] = (-1, 1)
    ) -> float:
        t = torch.linspace(domain[0], domain[1], 256, device=self.control_points.device)
        y = non_linearity(t)

        B = self.evaluate_basis(t)
        control_points:torch.Tensor = torch.linalg.pinv(B) @ y
        mse = (y - (B * control_points).sum(dim=-1)).square().mean()
        control_points = control_points.view(1, -1).repeat(self.out_features, 1)

        self.control_points.data = control_points
        return mse

    def compute_activations(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.offset) * self.scale

        B = self.evaluate_basis(x)
        w = torch.einsum('...id,oid->...oi', B, self.control_points)
        if self.bias is not None:
            w = w + self.bias
        return w
    