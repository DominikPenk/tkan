import copy
from typing import Callable

import torch

from .base import KanLinearBase, PolynomialKanLinear


def init_with_non_linearity(
    layer:KanLinearBase, 
    non_linearity:Callable[[torch.Tensor], torch.Tensor],
    domain:tuple[int, int]=(-1, 1)
) -> float:
    """
    Initializes a `KanLinearBase` layer with a given non-linearity function.

    Args:
        layer (KanLinearBase): The layer to be initialized.
        non_linearity (Callable[[torch.Tensor], torch.Tensor]): The non-linearity function used to initialize the layer.
        domain (tuple[int, int], optional): The domain of the input tensor. Defaults to (-1, 1).

    Returns:
        float: The mean squared error (MSE) of the best initialization found.

    Raises:
        TypeError: If the layer is not an instance of `KanLinearBase`.

    Notes:
        - If the layer supports the `init_with_non_linearity` method, it is called directly.
        - If the layer does not support the `init_with_non_linearity` method, the initialization is done via gradient descent.
    """
    if not isinstance(layer, KanLinearBase):
        raise TypeError(f"Layer must be of type KanLinearBase, got {type(layer)}")
    
    try:
        return layer.init_with_non_linearity(non_linearity, domain)
    except NotImplementedError:
        # Do it the hard way
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        
        t = torch.linspace(domain[0], domain[1], 256, device=next(layer.parameters()).device)
        t = t.view(-1, 1).repeat(1, layer.in_features)
        w_true = non_linearity(t)[:, None, :].repeat_interleave(layer.out_features, dim=1)

        patience = 3
        best_mse = float("inf")
        bad_updates = 0
        best_state = {}

        for _ in range(1500):
            w_pred = layer.compute_activations(t)
            loss = (w_true - w_pred).square().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss < best_mse:
                best_mse = loss
                best_state = copy.deepcopy(layer.state_dict())
                bad_updates = 0
            else:
                bad_updates += 1
                if bad_updates >= patience:
                    break

        layer.load_state_dict(best_state)
        return best_mse
