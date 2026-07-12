import numpy as np

from .base import TransitionKernel


class LogisticKernel(TransitionKernel):
    """
    Logistic transition kernel.

    The logistic function maps:

        (-inf, inf) -> (0, 1)

    Definition:

        K(x) = 1 / (1 + exp(-x))

    Derivative:

        K'(x) = K(x)(1-K(x))
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the logistic kernel.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Logistic kernel values.
        """

        return 1.0 / (1.0 + np.exp(-x))


    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the derivative of the logistic kernel.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Derivative values.
        """

        y = self.forward(x)

        return y * (1.0 - y)