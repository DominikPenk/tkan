from typing import Protocol, Literal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

##############
# Type hints #
##############

class NodeDrawer(Protocol):
    def __call__(
        self,
        x:np.ndarray,
        y:np.ndarray,
        size:float,
        ax:plt.Axes,
        transform:transforms.Transform
    ) -> None: 
        """Called once for each node in the network.
        
        Args:
            x (np.ndarray): The x-coordinates (in layer related coordinates) of the nodes.
            y (np.ndarray): The y-coordinates (in layer related coordinates) of the nodes.
            size (float): The size of the nodes (relative to default size).
            ax (plt.Axes): The axes on which to draw the nodes.
            transform (transforms.Transform): Transform used to convert layer related coordinates to NDC coordinates.
        """
        ...

    def finalize(
        self,
        ax:plt.Axes,
        transform:transforms.Transform
    ) -> None: ...

NodeTypeOptions = Literal['dots', 'squares']

class NodePositioner(Protocol):
    def __call__(
        self,
        num_nodes:int, 
        node_width:float,
        num_acts:int,
        act_width:float,
        outer_spacing:float,
        inner_spacing:float
    ) -> np.ndarray: ...

NodePositionOptions = Literal['default', 'compact', 'wide']

####################
# Node Positioning #
####################

def _get_default_node_positions(
    num_nodes:int,
    node_width:float,
    num_acts:int,
    act_width:float,
    outer_spacing:float,
    inner_spacing:float
) -> np.ndarray:
    required_width = num_nodes * node_width + (num_nodes - 1) * outer_spacing 
    left_most = (1.0 - required_width + node_width) * 0.5
    return left_most + np.arange(num_nodes) * (node_width + outer_spacing)

def _get_node_positions(
    num_nodes:int, 
    node_width:float,
    num_acts:int,
    act_width:float,
    outer_spacing:float,
    inner_spacing:float
) -> np.ndarray:
    return np.linspace(0.5 * node_width, 1.0 - 0.5 * node_width, num_nodes) if num_nodes > 1 else np.array([0.5])

def _get_compact_positions(
    num_nodes:int,
    node_width:float,
    num_acts:int,
    act_width:float,
    outer_spacing:float,
    inner_spacing:float
) -> np.ndarray:
    compact_node_width = num_acts * act_width + (num_acts - 1) * inner_spacing
    required_width = num_nodes * compact_node_width + (num_nodes - 1) * outer_spacing 
    left_most = (1.0 - required_width + compact_node_width) * 0.5
    return left_most + np.arange(num_nodes) * (compact_node_width + outer_spacing)
    
DEFAULT_NODE_POSITIONERS = {
    "default": _get_default_node_positions,
    "wide": _get_node_positions,
    "compact": _get_compact_positions
}

def get_node_positioner(type: str | NodePositioner) -> NodePositioner:
    """Return a NodePositioner based on the given type.
    If a string is given it must be a key in DEFAULT_NODE_POSITIONERS.

    Args:
        type (str | NodePositioner): The type of the node positioner to return.

    Returns:
        NodePositioner: The node positioner of the given type.
    """
    if isinstance(type, str):
        return DEFAULT_NODE_POSITIONERS[type]
    else:
        return type


################
# Node Drawing #
################

def _draw_dot_nodes(
    x:np.ndarray,
    y:np.ndarray,
    size:float,
    ax:plt.Axes,
    transform:transforms.Transform
) -> None:
    ax.scatter(x, y, transform=transform, color='black', s=size*30)    

def _draw_square_nodes(
    x:np.ndarray,
    y:np.ndarray,
    size:float,
    ax:plt.Axes,
    transform:transforms.Transform
) -> None:
    ax.scatter(x, y, transform=transform, color='black', marker='s', s=size*20)

DEFAULT_NODE_DRAWERS = {
    "dots": _draw_dot_nodes,
    'squares': _draw_square_nodes
}

def get_node_drawer(type: str | NodeDrawer) -> NodeDrawer:
    """Return a NodeDrawer based on the given type.
    If a string is given it must be a key in DEFAULT_NODE_DRAWERS.

    Args:
        type (str | NodeDrawer): The type of the node drawer to return.

    Returns:
        NodeDrawer: The node drawer of the given type.
    """
    if isinstance(type, str):
        return DEFAULT_NODE_DRAWERS[type]
    else:
        return type