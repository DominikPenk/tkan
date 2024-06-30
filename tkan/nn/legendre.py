import math

import torch
import torch.nn as nn

from .base import PolynomialKanLinear

LEGENDRE_POLYNOMIALS = [
    lambda x: torch.ones_like(x),
    lambda x: x,
    lambda x: 1.5 * x * x - 0.5,
    lambda x: x * (2.5 * x * x  - 1.5),
    lambda x: (3.0 + x * x * (35.0 * x * x - 30.0)) * 0.125,
    lambda x: (x * (15.0 + x * x * (63.0 * x * x -70.0))) * 0.125,
    lambda x: (x * x * (105.0 + x * x * (231.0 * x * x - 315.0)) - 5.0) * 0.0625,
    lambda x: (x * (x * x *(315.0 + x * x * (429.0 * x * x - 693.0)) - 35.0)) * 0.0625,
    lambda x: (x * x * (x * x * (x * x * (6435.0 * x * x - 12012.0) + 6930) - 1260.0) + 35.0) *  0.0078125,
    lambda x: (x * (x * x * (x * x * (x * x * (x * x * 12155.0 - 25740.0) + 18018.0) - 4620.0) + 315)) * 0.0078125,
    lambda x: (x * x * (x * x * (x * x * (x * x * (46189.0 * x * x - 109395.0) + 90090.0) - 30030.0) + 3465.0) - 63.0) * 0.00390625
]

def legendre_polynomials(x:torch.Tensor, order:int, min_order:int=0) -> torch.Tensor:
    Ps = [
        torch.ones_like(x),
        x
    ]
    for i in range(1, order):
        if i + 1 < len(LEGENDRE_POLYNOMIALS):
          Ps.append(LEGENDRE_POLYNOMIALS[i + 1](x))
        else:
          Ps.append(((2.0 * i + 1.0) * x * Ps[-1] - i * Ps[-2]) / (i + 1.0))
    return torch.stack(Ps[min_order:], dim=-1)


class LegendreKanLinear(PolynomialKanLinear):
    def get_initial_control_points(self) -> torch.Tensor:
        control_points = torch.empty(self.out_features, self.in_features, self.order).normal_(0, 0.1)
        sqrt_k = math.sqrt(1.0 / self.in_features)
        control_points[..., 0].uniform_(-sqrt_k, sqrt_k)
        return control_points


    def evaluate_basis(self, x:torch.Tensor) -> torch.Tensor:
        return legendre_polynomials(x, self.order, min_order=1)