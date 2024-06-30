import inspect
from typing import Callable

import torch


class TestFunction:
    def __init__(
        self,
        f: Callable
    ) -> None:
        self.dimensions = len(inspect.signature(f).parameters)
        self._f = f

    def __call__(self, *args: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if len(args) == 1 and self.dimensions > 1:
            args = torch.unbind(args[0], dim=-1)
        return self._f(*args)
    
    @torch.no_grad()
    def create_dataset(self, *sizes) -> tuple[torch.Tensor, ...]:
        n = sum(sizes)
        ys = torch.empty((0, ))
        xs = torch.empty((0, self.dimensions))
        while len(xs) < n:
            num_samples = n-len(xs)
            x = torch.empty((num_samples, self.dimensions)).uniform_(0.0, 1.0)
            y = self(x)
            mask = ~torch.isnan(y) & ~torch.isinf(y)
            xs = torch.cat([xs, x[mask]], dim=0)
            ys = torch.cat([ys, y[mask]], dim=0)

        ys = ys.unsqueeze(-1)
        result = tuple(zip(xs.split(sizes, dim=0), ys.split(sizes, dim=0)))
        if len(result) == 1:
            return result[0]
        return result

    