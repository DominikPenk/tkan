from typing import Literal, Protocol

import matplotlib.collections as collections
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np


class ConnectionDrawer(Protocol):
    def __call__(
        self,
        start:tuple[float, float],
        end:tuple[float, float],
        ax:plt.Axes,
        transform:transforms.Transform
    ) -> None: 
        """Called once for each connection in the network."""
        ...

    def finalize(
        self,
        ax:plt.Axes,
        transform:transforms.Transform
    ) -> None: 
        """Called once at the end of drawing a single layer."""
        ...

ConnectionTypeOptions = Literal['straight', 'split', 'curved']

class StraightConnectionDrawer:
    def __init__(self):
        self.connections = []

    def __call__(
        self,
        start:tuple[float, float],
        end:tuple[float, float],
        ax:plt.Axes,
        transform:transforms.Transform
    ) -> None:
        self.connections.append([start, end])

    def finalize(self, ax:plt.Axes, transform:transforms.Transform) -> None:
        ax.add_collection(collections.LineCollection(
            self.connections,
            transform=transform,
            color='black'
        ))
        self.connections = []

class SplitConnectionDrawer:
    def __init__(self, ratio:float=0.2):
        self.connections = []
        self.ratio = ratio

    def __call__(
        self,
        start:tuple[float, float],
        end:tuple[float, float],
        ax:plt.Axes,
        transform:transforms.Transform
    ) -> None:
        dy = end[1] - start[1]
        self.connections.append([
            start,
            [start[0], start[1] + dy * self.ratio],
            [end[0], end[1] - dy * self.ratio], 
            end
        ])

    def finalize(self, ax:plt.Axes, transform:transforms.Transform) -> None:
        ax.add_collection(collections.LineCollection(
            self.connections,
            transform=transform,
            color='black'
        ))
        self.connections = []

class CurvedConnectionDrawer:
    def __init__(self, easing:float=0.5):
        self.connections = []
        self.easing = easing


    @staticmethod
    def bezier_point(t, P0, P1, P2, P3):
        return (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3
    
    def __call__(
        self,
        start:tuple[float, float],
        end:tuple[float, float],
        ax:plt.Axes,
        transform:transforms.Transform
    ) -> None:
        dy = end[1] - start[1]
        P0 = np.asanyarray(start)
        P1 = np.asanyarray([start[0], start[1] + dy * self.easing])
        P2 = np.asanyarray([end[0], end[1] - dy * self.easing])
        P3 = np.asanyarray(end)

        ts = np.linspace(0, 1, 64).reshape(-1, 1)
        curve = self.bezier_point(ts, P0, P1, P2, P3)

        # self.connections.append(curve)
        self.connections.append(curve)


    def finalize(self, ax:plt.Axes, transform:transforms.Transform) -> None:
        ax.add_collection(collections.LineCollection(
            self.connections,
            transform=transform,
            color='black'
        ))
        self.connections = []

DEFAULT_CONNECTION_DRAWERS: dict[str, ConnectionDrawer] = {
    'straight': StraightConnectionDrawer(),
    'split': SplitConnectionDrawer(),
    'curved': CurvedConnectionDrawer()
}

def get_connection_drawer(type: str | ConnectionDrawer) -> ConnectionDrawer:
    """Return a ConnectionDrawer based on the given type.
    If a string is given it must be a key in DEFAULT_CONNECTION_DRAWERS.

    Args:
        type (str | ConnectionDrawer): The type of the connection drawer to return.

    Returns:
        ConnectionDrawer: The connection drawer of the given type.
    """
    if isinstance(type, str):
        return DEFAULT_CONNECTION_DRAWERS[type]
    else:
        return type
