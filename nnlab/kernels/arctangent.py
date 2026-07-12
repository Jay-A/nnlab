import numpy as np

from .base import TransitionKernel


class ArctangentKernel(TransitionKernel):
    """
    Arctangent transition kernel.

    This kernel is a normalized arctangent function.

    Definition:

        K(x) = 1/2 + atan(x)/pi

    Derivative:

        K'(x) = 1/(pi*(1+x^2))

    This transition has heavier tails than the logistic kernel.
    It corresponds to the cumulative distribution function of
    the standard Cauchy distribution.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the arctangent transition kernel.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Kernel values in the range (0, 1).
        """

        return 0.5 + np.arctan(x) / np.pi


    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the derivative of the arctangent kernel.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Derivative values.
        """

        return 1.0 / (np.pi * (1.0 + x**2))