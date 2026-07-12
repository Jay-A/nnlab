import numpy as np

from .base import TransitionKernel


class PolynomialKernel(TransitionKernel):
    """
    Polynomial smoothstep transition kernel.

    This kernel uses a cubic polynomial to create a smooth
    bounded transition.

    Definition:

        t = (x + 1) / 2

        K(x) = 3t^2 - 2t^3

    Derivative:

        K'(x) = 3t(1-t)

    The polynomial transition provides a smooth alternative
    to piecewise linear transitions with zero slope at the
    boundaries.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the polynomial transition kernel.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Kernel values in the range [0, 1].
        """

        t = np.clip((x + 1.0) / 2.0, 0.0, 1.0)

        return 3.0 * t**2 - 2.0 * t**3


    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the derivative of the polynomial kernel.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Derivative values.
        """

        t = np.clip((x + 1.0) / 2.0, 0.0, 1.0)

        return 3.0 * t * (1.0 - t)