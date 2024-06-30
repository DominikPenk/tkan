from typing import Any

import matplotlib.collections as collections
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ..nn.base import KanLinearBase
from .plotter import KanPlotter

def plot_kan(
    model:nn.Sequential | KanLinearBase,
    *,
    sample_inputs:torch.Tensor | None = None,
    ax:plt.Axes | None = None,
    **plotting_args
) -> plt.Figure:
    """
    Plots the KAN model using a KanPlotter.

    Args:
        model (nn.Sequential | KanLinearBase): The KAN model to be plotted.
        sample_inputs (torch.Tensor | None): Optional sample inputs to the model.
        ax (plt.Axes | None): Optional axes to plot on.
        **plotting_args: Additional keyword arguments for plotting.

    Returns:
        plt.Figure: The plotted figure.
    """
    plotter = KanPlotter(**plotting_args)
    return plotter(
        model, 
        sample_inputs=sample_inputs,
        ax=ax
    )