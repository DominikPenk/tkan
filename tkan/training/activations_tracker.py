from __future__ import annotations
from typing import Any
from functools import partial

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from ..nn.base import KanLinearBase


def _track_activations(
    layer:KanLinearBase,
    inputs:list[torch.Tensor],
    tracker:ActivationsTracker
) -> None:
    if tracker.track_always or layer.training:
        acts = layer.compute_activations(*inputs)
        tracker._activations[id(layer)] = acts

class ActivationsTracker(object):
    def __init__(
        self, 
        module:nn.Module | None = None,
        track_only_during_training:bool=True
    ) -> None:
        self._tracked_layers:list[KanLinearBase] = []
        self._hook_handles:dict[RemovableHandle]   = {}
        self._activations:dict[int, torch.Tensor]  = {}
        self.track_always = not track_only_during_training

        if module:
            self.add_module(module)

              
    def add_module(self, module:nn.Module, register_hooks:bool = False) -> None:
        for layer in module.modules():
            if isinstance(layer, KanLinearBase):
                self._tracked_layers.append(layer)

                if register_hooks:
                    self._hook_handles[id(layer)] = layer.register_forward_pre_hook(
                        partial(_track_activations, tracker=self)
                    )
    
    def register_hooks(self) -> None:
        for layer in self._tracked_layers:
            if id(layer) not in self._hook_handles:
                self._hook_handles[id(layer)] = layer.register_forward_pre_hook(
                    partial(_track_activations, tracker=self)
                )

    def unregister_hooks(self) -> None:
        for handle in self._hook_handles.values():
            handle.remove()
        self._hook_handles = {}

    def get_activations(self) -> list[torch.Tensor]:
        return list(self._activations.values())
    
    def get_activation(self, layer:KanLinearBase) -> torch.Tensor:
        if id(layer) not in self._activations:
            raise KeyError(f"Layer is not tracked")
        return self._activations[id(layer)]

    def reset(self) -> None:
        self._activations = {}

    def force_tracking(self, force:bool=True) -> ActivationsTracker:
        self.track_always = force
        return self

    @property
    def num_tracked_layers(self) -> int:
        return len(self._tracked_layers)

    def __enter__(self) -> None:
        self.register_hooks()

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        self.unregister_hooks()

