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
    
    The model also exposes trainable parameters owned
    by its constituent layers for use by optimizers.
    
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

    def parameters(
        self,
    ) -> list[np.ndarray]:
        """
        Return trainable parameters from all layers.

        Parameters are collected from each layer in model order.
        Layers without trainable parameters contribute an empty list.

        Returns
        -------
        list[np.ndarray]
            Collection of trainable model parameters.
        """

        parameters = []

        for layer in self.layers:
            parameters.extend(
                layer.parameters(),
            )

        return parameters

    def gradients(
        self,
    ) -> list[np.ndarray]:
        """
        Collect parameter gradients from all layers.
    
        Returns
        -------
        list[np.ndarray]
            Model gradients.
        """
    
        gradients = []
    
        for layer in self.layers:
            gradients.extend(
                layer.gradients()
            )
    
        return gradients