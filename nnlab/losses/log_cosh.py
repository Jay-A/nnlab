import numpy as np

from .base import Loss


class LogCoshLoss(Loss):
    """
    Log-cosh loss.

    The log-cosh loss computes the logarithm of the hyperbolic
    cosine of the prediction error.

    Definition:

        L = mean(log(cosh(prediction - target)))

    For small errors, the loss behaves like the mean squared
    error:

        log(cosh(x)) ≈ 0.5 * x^2

    For large errors, it behaves similarly to the mean absolute
    error up to an additive constant:

        log(cosh(x)) ≈ |x| - log(2)

    The log-cosh loss is smooth and differentiable everywhere,
    making it a robust alternative to both MSE and MAE.
    """

    def forward(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """
        Compute the mean log-cosh loss.

        Parameters
        ----------
        prediction : np.ndarray
            Model predictions.

        target : np.ndarray
            Target values.

        Returns
        -------
        float
            Mean log-cosh loss.
        """

        error = prediction - target

        # Numerically stable computation of log(cosh(x))
        loss = (
            np.abs(error)
            + np.log1p(np.exp(-2.0 * np.abs(error)))
            - np.log(2.0)
        )

        return float(np.mean(loss))

    def derivative(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the derivative of the log-cosh loss with respect
        to the predictions.

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

        return np.tanh(error) / prediction.size