from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """
    Abstract base class for neural network models.

    A model defines a complete computation graph
    composed of one or more layers.

    Models provide two complementary operations:

    forward:
        Computes model predictions from input data.

    backward:
        Propagates gradients from the loss back through
        the computation graph.
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

    @abstractmethod
    def backward(
        self,
        gradient: np.ndarray,
    ) -> np.ndarray:
        """
        Compute model backward pass.

        The backward pass propagates gradients through the
        computation graph using the chain rule.

        Parameters
        ----------
        gradient : np.ndarray
            Gradient of loss with respect to model output.

        Returns
        -------
        np.ndarray
            Gradient of loss with respect to model input.
        """
        pass