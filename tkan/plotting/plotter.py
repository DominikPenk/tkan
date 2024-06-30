from typing import Optional, Protocol

import matplotlib.collections as collections
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import torch
import torch.nn as nn

from ..nn.base import KanLinearBase
from ..training import ActivationsTracker
from .connections import ConnectionDrawer, get_connection_drawer, ConnectionTypeOptions
from .nodes import (NodeDrawer, NodePositioner, NodePositionOptions,
                    NodeTypeOptions, get_node_drawer, get_node_positioner)


def _get_aspect_ratio(ax:plt.Axes):
    aspect = ax.get_aspect()
    if aspect == 'auto':
        # Get the data limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = abs(xlim[1] - xlim[0])
        y_range = abs(ylim[1] - ylim[0])
        
        # Get the size of the figure in inches
        fig_width, fig_height = ax.get_figure().get_size_inches()
        
        # Get the size of the axis in the figure (inches)
        bbox = ax.get_position()
        axis_width = bbox.width * fig_width
        axis_height = bbox.height * fig_height
        
        # Calculate the aspect ratio
        aspect_ratio = (y_range / x_range) * (axis_width / axis_height)
        return aspect_ratio
    elif aspect == 'equal':
        return 1.0
    return aspect

class KanPlotter:
    def __init__(
        self,
        *,
        connection_type:ConnectionTypeOptions | ConnectionDrawer = 'straight',
        node_type:NodeTypeOptions | NodeDrawer = 'dots',
        node_positions:NodePositionOptions | NodePositioner = 'default',
        node_size:float = 1.0,
        figure_padding:float = 0.025, 
        inner_act_spacing:float = 0.05,
        outer_act_spacing:float = 0.1
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            connection_type (ConnectionTypeOptions | ConnectionDrawer, optional): The type of connection to use. Defaults to 'straight'.
            node_type (NodeTypeOptions | NodeDrawer, optional): The type of node to use. Defaults to 'dots'.
            node_positions (NodePositionOptions | NodePositioner, optional): The position of the nodes. Defaults to 'default'.
            node_size (float, optional): The size of the nodes. Defaults to 1.0.
            figure_padding (float, optional): The padding around the figure. Defaults to 0.025.
            inner_act_spacing (float, optional): The spacing between inner activations. Defaults to 0.05.
            outer_act_spacing (float, optional): The spacing between outer activations. Defaults to 0.1.

        """
        self.connection_type = connection_type
        self.node_type = node_type
        self.node_positions = node_positions
        self.figure_padding = figure_padding
        self.inner_act_spacing = inner_act_spacing
        self.outer_act_spacing = outer_act_spacing
        self.node_size = node_size

    def __call__(
        self, 
        model:nn.Sequential | KanLinearBase,
        *,
        sample_inputs:Optional[torch.Tensor] = None,
        ax:Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Visualize structure and activations of a KAN model.

        Args:
            model (nn.Sequential | KanLinearBase): The model or single KAN layer to visualize.
            sample_inputs (Optional[torch.Tensor], optional): The input tensor to use for tracking activations. Defaults to None.
            ax (Optional[plt.Axes], optional): The matplotlib axes to plot on. If None, the current axes will be used. Defaults to None.

        Returns:
            plt.Figure: The matplotlib figure containing the plot.

        Raises:
            ValueError: If the model is not a nn.Sequential, KAN layer or if it contains layers that are not Kan-layers.
        """
        if isinstance(model, KanLinearBase):
            model = nn.Sequential(model)

        # Check that model is a Sequential model with only Kan layers
        if not isinstance(model, nn.Sequential):
            raise ValueError("Model must be a nn.Sequential")
        
        if not all(isinstance(layer, KanLinearBase) for layer in model):
            raise ValueError("All layers must be Kan-layers")

        ax = ax or plt.gca()
        fig = ax.get_figure()

        aspect_ratio = _get_aspect_ratio(ax)


        transBase = transforms.Affine2D() \
            .scale(1.0-self.figure_padding, 1.0-self.figure_padding) \
            .translate(self.figure_padding / 2, self.figure_padding / 2) + ax.transAxes


        act_width = 1.0 / max([
            layer.in_features*layer.out_features + 
            self.inner_act_spacing * layer.out_features * (layer.in_features - 1) +
            self.outer_act_spacing * (layer.out_features - 1) 
            for layer in model
        ])
        act_height = act_width * aspect_ratio

        inner_spacing = act_width * self.inner_act_spacing
        outer_spacing = act_width * self.outer_act_spacing

        max_outputs = max(layer.out_features for layer in model)
        node_width  = act_width * max_outputs + inner_spacing * (max_outputs - 1)

        line_height = 1.0 / len(model)

        connection_drawer = self.connection_type
        node_drawer       = self.node_type
        node_positioner   = self.node_positions

        if sample_inputs is not None:
            tracker = ActivationsTracker(model)
            with tracker.force_tracking():
                model(sample_inputs.view(-1, sample_inputs.size(-1)))
        else:
            tracker = None

        for layer_id, layer in enumerate(model):
            transLayer = transforms.Affine2D().translate(0, line_height * layer_id) + transBase

            # Get node positions
            in_node_positions  = node_positioner(
                layer.in_features,  
                node_width=node_width,
                num_acts=layer.out_features,
                act_width=act_width,
                outer_spacing=outer_spacing, 
                inner_spacing=inner_spacing
            )
            out_node_positions = node_positioner(
                layer.out_features, 
                node_width=node_width, 
                num_acts=model[layer_id + 1].out_features if layer_id < len(model) - 1 else layer.out_features,
                act_width=act_width, 
                outer_spacing=outer_spacing, 
                inner_spacing=inner_spacing
            )
            # Draw them
            node_drawer(
                x=in_node_positions,
                y=np.zeros_like(in_node_positions),
                size=self.node_size,
                ax=ax,
                transform=transLayer
            )
            if layer_id == len(model) - 1:
                node_drawer(
                    x=out_node_positions,
                    y=np.full_like(out_node_positions, line_height),
                    size=self.node_size,
                    ax=ax,
                    transform=transLayer
                )

            width_activation_area = layer.out_features * act_width + (layer.out_features - 1) * inner_spacing
            # Calculate left edges of activation plots shape (num_inputs, num_outputs) 
            act_positions = in_node_positions[:, None] - 0.5 * width_activation_area + np.arange(layer.out_features) * (act_width + inner_spacing)
            
            if tracker is not None:
                input_values = tracker.get_activation(model[layer_id - 1]).sum(dim=-1) if layer_id > 0 else sample_inputs

            # Create activation areas
            for activation_id in range(layer.in_features*layer.out_features):
                in_id, out_id = divmod(activation_id, layer.out_features)
                
                #  Create the new axes
                transAct = transforms.Affine2D().translate(
                    act_positions[in_id, out_id],
                    0.5 * (line_height - act_height)
                ) + transLayer

                transActivationAxes = transAct + fig.transSubfigure.inverted()
                (ax_left, ax_bottom), (ax_right, ax_top) = transActivationAxes.transform(((0, 0), (act_width, act_height)))
                new_ax = fig.add_axes([ax_left, ax_bottom, ax_right-ax_left, ax_top-ax_bottom])
                new_ax.set_xticks([])
                new_ax.set_yticks([])

                # Actually plot the activation
                if tracker is not None:
                    trange = (
                        input_values[:, in_id].min().item(),
                        input_values[:, in_id].max().item()
                    )
                else:
                    trange = None
                layer.plot_activation(
                    activation_id=(out_id, in_id), 
                    ax=new_ax,
                    trange=trange
                )

                # Plot connections
                connection_drawer(
                    start=(
                        in_node_positions[in_id], 
                        0
                    ),
                    end=(
                        act_positions[in_id, out_id] + 0.5 * act_width, 
                        0.5 * (line_height - act_height)
                    ),
                    ax=ax,
                    transform=transLayer
                )
                connection_drawer(
                    start=(
                        act_positions[in_id, out_id] + 0.5 * act_width, 
                        line_height - 0.5 * (line_height - act_height)
                    ),
                    end=(
                        out_node_positions[out_id], 
                        line_height
                    ),
                    ax=ax,
                    transform=transLayer,
                )

            # Finalize the drawers if neccessary
            if hasattr(connection_drawer, 'finalize'):
                connection_drawer.finalize(ax=ax, transform=transLayer)
            if hasattr(node_drawer, 'finalize'):
                node_drawer.finalize(ax=ax, transform=transLayer)
                
        ax.set_axis_off()
        # Set current axis to ax
        plt.sca(ax)
        return fig


    @property
    def connection_type(self) -> ConnectionDrawer:
        return get_connection_drawer(self._connection_type)
    
    @connection_type.setter
    def connection_type(self, type: str | ConnectionDrawer) -> None:
        self._connection_type = type

    @property
    def node_type(self) -> NodeDrawer:
        return get_node_drawer(self._node_type)
    
    @node_type.setter
    def node_type(self, type: str | NodeDrawer) -> None:
        self._node_type = type

    @property
    def node_positions(self) -> NodePositioner:
        return get_node_positioner(self._node_positions)
    
    @node_positions.setter
    def node_positions(self, type: str | NodePositioner) -> None:
        self._node_positions = type
        