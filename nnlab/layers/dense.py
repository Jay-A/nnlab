import numpy as np

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
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        """
        Initialize dense layer parameters.

        Parameters
        ----------
        input_size : int
            Number of input features.

        output_size : int
            Number of output features.
        """

        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(
            input_size,
            output_size,
        ) * np.sqrt(2.0 / input_size)

        self.bias = np.zeros(
            output_size,
        )

        # Cached input from forward pass.
        # Required to compute parameter gradients.
        self.input = None

        # Gradients computed during backward pass.
        # These will later be consumed by optimizers.
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