from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """
    Abstract base class for neural network models.

    A model defines a complete computation graph
    composed of one or more layers.
    """

    @abstractmethod
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
        pass