import numpy as np

from .base import TransitionKernel


class GaussianRBFKernel(TransitionKernel):
    """
    Gaussian radial basis function kernel, 
    not really a transition kernel.

    This kernel produces a localized response centered around zero.

    Definition:

        K(x) = exp(-x^2 / 2)

    Derivative:

        K'(x) = -x * exp(-x^2 / 2)

    Unlike sigmoid-like transition kernels, the Gaussian RBF
    approaches zero at both positive and negative infinity.

    This kernel is useful for localized feature responses,
    radial basis networks, and function approximation.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Gaussian RBF kernel.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Kernel response values in the range (0, 1].
        """

        return np.exp(-0.5 * x**2)


    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the derivative of the Gaussian RBF kernel.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Derivative values.
        """

        return -x * np.exp(-0.5 * x**2)