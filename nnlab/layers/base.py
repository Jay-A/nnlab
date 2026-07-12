from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    """
    Abstract base class for neural network layers.

    Layers participate in a computational graph through two
    complementary operations:

    forward:
        Computes activations during inference.

    backward:
        Propagates gradients during optimization.

    A layer does not update parameters itself. Parameter updates
    are handled separately by optimizer components.
    """

    @abstractmethod
    def forward(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Compute forward pass.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Layer output.
        """
        pass

    @abstractmethod
    def backward(
        self,
        gradient: np.ndarray,
    ) -> np.ndarray:
        """
        Compute backward pass.

        Parameters
        ----------
        gradient : np.ndarray
            Gradient of loss with respect to layer output.

        Returns
        -------
        np.ndarray
            Gradient of loss with respect to layer input.
        """
        pass