import numpy as np

from .base import Layer


import numpy as np

from ..initializers import HeNormal
from .base import Layer


class Dense(Layer):
    """
    Fully connected linear layer.

    The dense layer performs an affine transformation:

        y = x @ W + b

    During the forward pass, the input is stored so that gradients
    can be computed during the backward pass.

    Parameters
    ----------
    input_size : int
        Number of input features.

    output_size : int
        Number of output features.

    initializer : Initializer, optional
        Parameter initialization strategy used for weights.
        Defaults to HeNormal.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        initializer=None,
    ):
        """
        Initialize dense layer parameters.

        Parameters
        ----------
        input_size : int
            Number of input features.

        output_size : int
            Number of output features.

        initializer : Initializer, optional
            Weight initialization strategy.
        """

        self.input_size = input_size
        self.output_size = output_size

        if initializer is None:
            initializer = HeNormal()

        self.initializer = initializer

        self.weights = self.initializer.initialize(
            shape=(
                input_size,
                output_size,
            ),
            fan_in=input_size,
            fan_out=output_size,
        )

        self.bias = np.zeros(
            output_size,
        )

        self.input = None

        self.weight_gradient = None
        self.bias_gradient = None

    def forward(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Compute forward transformation.

        The forward pass computes:

            y = x @ W + b

        The input is cached for use during backpropagation.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Transformed values.
        """

        self.input = x

        return x @ self.weights + self.bias

    def backward(
        self,
        gradient: np.ndarray,
    ) -> np.ndarray:
        """
        Compute backward gradient propagation.

        Given the gradient of the loss with respect to the
        layer output:

            dL/dy

        compute:

            dL/dW
            dL/db
            dL/dx

        Parameters
        ----------
        gradient : np.ndarray
            Gradient of loss with respect to layer output.

        Returns
        -------
        np.ndarray
            Gradient of loss with respect to layer input.
        """

        self.weight_gradient = (
            self.input.T @ gradient
        )

        self.bias_gradient = np.sum(
            gradient,
            axis=0,
        )

        return gradient @ self.weights.T

    def parameters(
        self,
    ) -> list[np.ndarray]:
        """
        Return trainable parameters.
    
        Returns
        -------
        list[np.ndarray]
            Dense layer parameters.
        """
    
        return [
            self.weights,
            self.bias,
        ]

    def gradients(
        self,
    ) -> list[np.ndarray]:
        """
        Return gradients associated with trainable parameters.

        The returned gradients correspond to the parameters
        returned by ``parameters()`` in the same order.

        Returns
        -------
        list[np.ndarray]
            Dense layer parameter gradients.
        """

        return [
            self.weight_gradient,
            self.bias_gradient,
        ]        