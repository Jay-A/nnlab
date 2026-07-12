import numpy as np

from .base import Loss


class MeanAbsoluteError(Loss):
    """
    Mean absolute error (MAE) loss.

    The mean absolute error measures the average absolute
    difference between model predictions and target values.

    Definition:

        L = mean(abs(prediction - target))

    The MAE is less sensitive to outliers than the mean
    squared error but is not differentiable at zero.
    """

    def forward(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """
        Compute the mean absolute error.

        Parameters
        ----------
        prediction : np.ndarray
            Model predictions.

        target : np.ndarray
            Target values.

        Returns
        -------
        float
            Mean absolute error.
        """

        error = prediction - target

        return float(np.mean(np.abs(error)))

    def derivative(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the derivative of the mean absolute error
        with respect to the predictions.

        Parameters
        ----------
        prediction : np.ndarray
            Model predictions.

        target : np.ndarray
            Target values.

        Returns
        -------
        np.ndarray
            Subgradient of the loss.
        """

        n = prediction.size

        return np.sign(prediction - target) / n