import numpy as np

from .base import Model
from ..layers import Layer


class FeedForward(Model):
    """
    Feed-forward neural network model.

    A feed-forward model applies a sequence of layers
    in order:

        x -> layer1 -> layer2 -> ... -> output

    Parameters
    ----------
    layers : list[Layer]
        Ordered list of layers.
    """

    def __init__(
        self,
        layers: list[Layer],
    ):
        self.layers = layers

    def forward(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Compute model output.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Model output.
        """

        for layer in self.layers:
            x = layer.forward(x)

        return x