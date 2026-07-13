import numpy as np

from .base import Optimizer


class SGD(Optimizer):
    """
    Stochastic gradient descent optimizer.

    SGD updates parameters according to:

        parameter = parameter - learning_rate * gradient

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size used during parameter updates.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
    ):
        """
        Initialize stochastic gradient descent.

        Parameters
        ----------
        learning_rate : float, default=0.01
            Step size used for updates.

        Raises
        ------
        ValueError
            If learning_rate is not positive.
        """

        if learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive."
            )

        self.learning_rate = learning_rate

    def step(
        self,
        parameters: list[np.ndarray],
        gradients: list[np.ndarray],
    ) -> None:
        """
        Update parameters using gradients.

        Parameters
        ----------
        parameters : list[np.ndarray]
            Trainable parameters to update.

        gradients : list[np.ndarray]
            Gradients corresponding to each parameter.

        Returns
        -------
        None
            Parameters are updated in place.
        """

        for parameter, gradient in zip(
            parameters,
            gradients,
        ):
            parameter -= (
                self.learning_rate
                * gradient
            )