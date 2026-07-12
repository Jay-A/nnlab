import numpy as np

from ..activations import Activation
from .base import Layer


class ActivationLayer(Layer):
    """
    Neural network layer wrapper for activation functions.

    This layer allows activation functions to participate in a model
    computation graph alongside other layers such as Dense layers.

    Parameters
    ----------
    activation : Activation
        Activation function applied to the layer input.
    """

    def __init__(
        self,
        activation: Activation,
    ):
        """
        Initialize activation layer.

        Parameters
        ----------
        activation : Activation
            Activation function used during the forward pass.
        """

        self.activation = activation

    def forward(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Apply activation function to input values.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Transformed output values.
        """

        return self.activation.forward(x)