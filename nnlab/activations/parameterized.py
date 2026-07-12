import numpy as np

from nnlab.kernels import TransitionKernel

from .base import Activation


class ParameterizedActivation(Activation):
    """
    Activation function constructed from a parameterized kernel.

    The activation is defined as:

        phi(x) = a * K((x - c) / s) + b

    where:

        K : transition kernel
        c : center location
        s : kernel input scaling factor
        a : amplitude
        b : output offset

    This class separates the mathematical shape of an activation
    from its configurable behavior.
    """

    def __init__(
        self,
        kernel: TransitionKernel,
        center: float = 0.0,
        kernel_scale: float = 1.0,
        amplitude: float = 1.0,
        bias: float = 0.0,
    ):
        """
        Initialize a parameterized activation.

        Parameters
        ----------
        kernel : TransitionKernel
            Mathematical kernel defining the activation shape.

        center : float
            c in activation function.
            Controls the horizontal location of the activation
            transition.

        kernel_scale : float
            s in activation function.
            Controls horizontal scaling of the kernel input.
            Larger values produce broader, smoother transitions;
            smaller values produce sharper transitions.

        amplitude : float
            a in activation function.
            Controls vertical scaling of the activation output.

        bias : float
            b in activation function.
            Controls vertical offset of the activation output.
        """

        if kernel_scale <= 0:
            raise ValueError("Activation kernel_scale must be positive.")

        self.kernel = kernel
        self.center = center
        self.kernel_scale = kernel_scale
        self.amplitude = amplitude
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

        z = (x - self.center) / self.kernel_scale

        return self.amplitude * self.kernel.forward(z) + self.bias

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

        z = (x - self.center) / self.kernel_scale

        return (
            self.amplitude
            * self.kernel.derivative(z)
            / self.kernel_scale
        )