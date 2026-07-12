import numpy as np

from .base import TransitionKernel


class PiecewiseLinearKernel(TransitionKernel):
    """
    Piecewise linear transition kernel.

    This kernel provides a bounded linear transition between
    two constant regions.

    Definition:

        K(x) =
            0              x <= -1
            (x + 1) / 2    -1 < x < 1
            1              x >= 1

    Derivative:

        K'(x) =
            0              |x| >= 1
            1/2            |x| < 1

    This kernel provides a simple connection between smooth
    transition functions and hard piecewise activations.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the piecewise linear kernel.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Kernel values in the range [0, 1].
        """

        return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the derivative of the piecewise linear kernel.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Derivative values.
        """

        return np.where(np.abs(x) < 1.0, 0.5, 0.0)