import math

import torch
import torch.nn as nn
from .base import PolynomialKanLinear


def hermite_polynomials(x:torch.Tensor, n:int, min_order:int = 0) -> torch.Tensor:
    Ps = [
        torch.ones_like(x),
        x
    ]
    for i in range(1, n):
        Ps.append(x * Ps[-1] - i * Ps[-2])
    return torch.stack(Ps[min_order:], dim=-1)


class HermiteKanLinear(PolynomialKanLinear):
    def get_initial_control_points(self) -> torch.Tensor:
        control_points = torch.zeros(self.out_features, self.in_features, self.order)
        sqrt_k = math.sqrt(1.0 / self.in_features)
        control_points[..., 0].uniform_(-sqrt_k, sqrt_k)
        return control_points


    def evaluate_basis(self, x:torch.Tensor) -> torch.Tensor:
        return hermite_polynomials(x, self.order, min_order=1)