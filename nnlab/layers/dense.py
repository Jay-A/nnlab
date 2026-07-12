import numpy as np

from .base import Layer


class Dense(Layer):
    """
    Fully connected linear layer.

    Computes:

        y = Wx + b

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
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(
            input_size,
            output_size,
        ) * np.sqrt(2.0 / input_size)

        self.bias = np.zeros(output_size)

    def forward(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Compute linear transformation.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Transformed values.
        """

        return x @ self.weights + self.bias