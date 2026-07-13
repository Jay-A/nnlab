from abc import ABC, abstractmethod

import numpy as np


class Initializer(ABC):
    """
    Abstract base class for parameter initialization strategies.

    Initializers construct the initial values of trainable parameters
    before optimization begins.

    Initialization strategies influence optimization dynamics,
    convergence behavior, and learning stability. Separating initialization
    from layers allows initialization strategies to be studied independently
    as experimental components.

    Subclasses implement ``initialize()`` to generate parameter arrays.
    """

    @abstractmethod
    def initialize(
        self,
        shape: tuple[int, ...],
        fan_in: int,
        fan_out: int,
    ) -> np.ndarray:
        """
        Generate an initialized parameter array.

        Parameters
        ----------
        shape : tuple[int, ...]
            Desired shape of the parameter array.

        fan_in : int
            Effective number of input units contributing to each output.

        fan_out : int
            Effective number of output units receiving input from each
            parameter.

        Returns
        -------
        np.ndarray
            Initialized parameter array.
        """
        pass