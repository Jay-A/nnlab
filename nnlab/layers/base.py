from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    """
    Abstract base class for neural network layers.
    """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
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