from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    """
    Abstract base class for loss functions.

    A loss function quantifies the discrepancy between model
    predictions and target values. Losses are used to evaluate
    model performance and, during training, provide the objective
    minimized by optimization algorithms.
    """

    @abstractmethod
    def forward(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """
        Compute the loss value.

        Parameters
        ----------
        prediction : np.ndarray
            Model predictions.

        target : np.ndarray
            Target values.

        Returns
        -------
        float
            Scalar loss value.
        """
        pass

    @abstractmethod
    def derivative(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the derivative of the loss with respect to the
        model predictions.

        Parameters
        ----------
        prediction : np.ndarray
            Model predictions.

        target : np.ndarray
            Target values.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to the predictions.
        """
        pass