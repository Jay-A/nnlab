import numpy as np

from .base import Loss


class HuberLoss(Loss):
    """
    Huber loss.

    The Huber loss behaves quadratically for small prediction
    errors and linearly for large errors.

    Definition:

        L(e) = 0.5 * e^2                     if |e| <= delta
             = delta * (|e| - 0.5 * delta)   otherwise

    where

        e = prediction - target

    The Huber loss combines the smoothness of the mean squared
    error with the robustness of the mean absolute error.

    Parameters
    ----------
    delta : float, default=1.0
        Threshold separating quadratic and linear regions.
    """

    def __init__(
        self,
        delta: float = 1.0,
    ):
        """
        Initialize the Huber loss.

        Parameters
        ----------
        delta : float, default=1.0
            Transition threshold.
        """

        if delta <= 0.0:
            raise ValueError("delta must be positive.")

        self.delta = delta

    def forward(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """
        Compute the Huber loss.

        Parameters
        ----------
        prediction : np.ndarray
            Model predictions.

        target : np.ndarray
            Target values.

        Returns
        -------
        float
            Mean Huber loss.
        """

        error = prediction - target

        quadratic = np.abs(error) <= self.delta

        loss = np.where(
            quadratic,
            0.5 * error**2,
            self.delta * (
                np.abs(error) - 0.5 * self.delta
            ),
        )

        return float(np.mean(loss))

    def derivative(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the derivative of the Huber loss.

        Parameters
        ----------
        prediction : np.ndarray
            Model predictions.

        target : np.ndarray
            Target values.

        Returns
        -------
        np.ndarray
            Gradient of the loss.
        """

        error = prediction - target

        gradient = np.where(
            np.abs(error) <= self.delta,
            error,
            self.delta * np.sign(error),
        )

        return gradient / prediction.size