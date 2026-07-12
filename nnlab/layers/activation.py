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

        # Cached input from forward pass.
        # Required for computing the activation derivative
        # during the backward pass.
        self.input = None

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

        self.input = x

        return self.activation.forward(x)

    def backward(
        self,
        gradient: np.ndarray,
    ) -> np.ndarray:
        """
        Compute gradient through activation function.

        Uses the chain rule:

            dL/dx = dL/dy * dy/dx

        where:

            dL/dy is the incoming gradient
            dy/dx is the activation derivative

        Parameters
        ----------
        gradient : np.ndarray
            Gradient of loss with respect to layer output.

        Returns
        -------
        np.ndarray
            Gradient of loss with respect to layer input.
        """

        return (
            gradient
            * self.activation.derivative(self.input)
        )