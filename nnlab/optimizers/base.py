from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    """
    Abstract base class for optimization algorithms.

    Optimizers update model parameters using gradients
    computed during backpropagation.

    An optimizer does not compute gradients. It only applies
    updates to parameters provided by a model.
    """

    @abstractmethod
    def step(
        self,
        parameters: list[np.ndarray],
        gradients: list[np.ndarray],
    ) -> None:
        """
        Update model parameters.

        Parameters
        ----------
        parameters : list[np.ndarray]
            Trainable model parameters.

        gradients : list[np.ndarray]
            Gradients associated with the parameters.

        Returns
        -------
        None
            Parameters are updated in place.
        """
        pass