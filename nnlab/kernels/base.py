from abc import ABC, abstractmethod

import numpy as np


class TransitionKernel(ABC):
    """
    Base class for transition kernels.

    A kernel defines the mathematical shape used
    to construct parameterized activation functions.
    """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the kernel.
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the kernel derivative.
        """
        pass