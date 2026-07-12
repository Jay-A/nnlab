import numpy as np

from nnlab.kernels import TransitionKernel

from .base import Activation


class ParameterizedActivation(Activation):
    """
    Activation function constructed from a parameterized kernel.

    The activation is defined as:

        phi(x) = a * K((x - c) / w) + b

    where:

        K : transition kernel
        c : center location
        w : transition width
        a : output scale
        b : output offset

    This class separates the mathematical shape of an activation
    from its configurable behavior.
    """

    def __init__(
        self,
        kernel: TransitionKernel,
        center: float = 0.0,
        width: float = 1.0,
        scale: float = 1.0,
        bias: float = 0.0,
    ):
        """
        Initialize a parameterized activation.

        Parameters
        ----------
        kernel : TransitionKernel
            Mathematical kernel defining the activation shape.

        center : float
            Location of the activation transition.

        width : float
            Controls transition sharpness.

        scale : float
            Output scaling factor.

        bias : float
            Output offset.
        """

        if width <= 0:
            raise ValueError("Activation width must be positive.")

        self.kernel = kernel
        self.center = center
        self.width = width
        self.scale = scale
        self.bias = bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the activation function.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Activated output values.
        """

        z = (x - self.center) / self.width

        return self.scale * self.kernel.forward(z) + self.bias

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the activation derivative.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Derivative values.
        """

        z = (x - self.center) / self.width

        return (
            self.scale
            * self.kernel.derivative(z)
            / self.width
        )