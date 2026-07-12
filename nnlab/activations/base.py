from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    """
    Abstract base class for neural network activation functions.

    Activations transform neuron inputs into nonlinear outputs
    and provide derivatives required for optimization.
    """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the activation function.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Activated output values.
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the activation derivative.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Derivative values used during optimization.
        """
        pass