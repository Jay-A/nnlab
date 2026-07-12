import numpy as np
from scipy.special import erf

from .base import TransitionKernel


class GaussianCDFKernel(TransitionKernel):
    """
    Gaussian cumulative distribution transition kernel.

    This kernel is the cumulative distribution function of the
    standard normal distribution.

    Definition:

        K(x) = 1/2 * (1 + erf(x / sqrt(2)))

    Derivative:

        K'(x) = exp(-x^2 / 2) / sqrt(2*pi)

    The Gaussian CDF provides a smooth sigmoid-like transition
    with rapidly decaying tails.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Gaussian CDF kernel.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Kernel values in the range (0, 1).
        """

        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the derivative of the Gaussian CDF kernel.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Derivative values.
        """

        return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)