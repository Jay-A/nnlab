import numpy as np

from ..layers import Layer
from .base import Model


class FeedForward(Model):
    """
    Feed-forward neural network model.

    A feed-forward model applies a sequence of layers
    in order:

        x -> layer1 -> layer2 -> ... -> output

    During the backward pass, gradients are propagated
    through the same layers in reverse order:

        gradient -> layerN -> ... -> layer2 -> layer1

    Parameters
    ----------
    layers : list[Layer]
        Ordered list of layers.
    """

    def __init__(
        self,
        layers: list[Layer],
    ):
        """
        Initialize feed-forward model.

        Parameters
        ----------
        layers : list[Layer]
            Ordered list of layers composing the model.
        """

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

    def backward(
        self,
        gradient: np.ndarray,
    ) -> np.ndarray:
        """
        Compute model backward pass.

        Gradients are propagated through layers in reverse
        order using the chain rule.

        Parameters
        ----------
        gradient : np.ndarray
            Gradient of loss with respect to model output.

        Returns
        -------
        np.ndarray
            Gradient of loss with respect to model input.
        """

        for layer in reversed(self.layers):
            gradient = layer.backward(
                gradient,
            )

        return gradient