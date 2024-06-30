import torch

import torch.nn as nn

from .activations_tracker import ActivationsTracker
from ..nn.base import KanLinearBase

def prune_from_data(
    model:nn.Sequential,
    data:torch.Tensor,
    theta:float=1e-2,
) -> nn.Sequential:
    """
    Prunes the given sequential model consisting of Kan layers by removing nodes with small incoming and outgoing activations. 

    Args:
        model (nn.Sequential): The model to be pruned. Must be a nn.Sequential containing only Kan layers.
        data (torch.Tensor): The data used to compute activations for pruning.
        theta (float, optional): The threshold value for pruning. Defaults to 1e-2.

    Returns:
        nn.Sequential: The pruned model.

    Raises:
        ValueError: If the model is not a nn.Sequential or if any layer is not a KanLinearBase.
    """
        
    # Make sure all layers are Kan-layers
    if not isinstance(model, nn.Sequential):
        raise ValueError("Module must be a nn.Sequential")
    
    if not all(isinstance(layer, KanLinearBase) for layer in model):
        raise ValueError("All layers must be Kan-layers")
    

    tracker = ActivationsTracker(model)
    with tracker.force_tracking():
        model.eval()
        model(data.view(-1, data.size(-1)))

    activations = [
        tracker.get_activation(layer)
        for layer in model
    ]

    return prune_from_activations(
        model=model,
        activations=activations,
        theta=theta
    )


def prune_from_activations(
    model:nn.Sequential,
    activations:list[torch.Tensor],
    theta:float=1e-2,
) -> nn.Sequential:
    # Make sure all layers are Kan-layers
    if not isinstance(model, nn.Sequential):
        raise ValueError("Module must be a nn.Sequential")
    
    if not all(isinstance(layer, KanLinearBase) for layer in model):
        raise ValueError("All layers must be Kan-layers")
    
    # Ensure that activations match the model
    if len(activations) != len(model):
        raise ValueError("Number of activations must match number of layers in the model")
    for layer_id, (act, layer) in enumerate(zip(activations, model)):
        if act.shape[-2:] != (layer.out_features, layer.in_features):
            raise ValueError(f"Activation shape {act.shape} in layer {layer_id} does not match expected shape (*, {layer.out_features}, {layer.in_features})")
        

    # Reshape activations to have a single batch dimension
    activations = [act.view(-1, act.size(-2), act.size(-1)) for act in activations]
    
    active_nodes = []
    for layer_id in range(len(model) - 1):
        layer:KanLinearBase = model[layer_id]
        if layer.out_features == 1:
            continue
        L1_in:torch.Tensor  = activations[layer_id].abs().mean(dim=0).max(dim=-1).values
        L1_out:torch.Tensor = activations[layer_id+1].abs().mean(dim=0).max(dim=0).values

        prune_ids = torch.nonzero((L1_in < theta) | (L1_out < theta)).flatten()
        if len(prune_ids):
            # Nodes should be pruned in order of lowest L1 norm
            prune_ids = prune_ids[torch.argsort(L1_out[prune_ids])]
            # Make sure we have at least 1 node left
            prune_ids = prune_ids[:layer.out_features - 1]

        active_nodes.append([i for i in range(layer.out_features) if i not in prune_ids])
    
    # We cannot remove nodes from the final (output) layer
    active_nodes.append([i for i in range(model[-1].out_features)])

    pruned_layers:list[KanLinearBase] = []
    for layer_id, layer in enumerate(model):
        out_features = active_nodes[layer_id] if layer_id < len(model) - 1 else [i for i in range(layer.out_features)]
        in_features = active_nodes[layer_id - 1] if layer_id > 0 else [i for i in range(layer.in_features)]
        pruned_layers.append(layer.get_pruned(in_features, out_features)) 
    
    return nn.Sequential(*pruned_layers).to(next(model.parameters()).device)