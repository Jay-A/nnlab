import numpy as np

from .base import Initializer


class HeNormal(Initializer):
    """
    He normal parameter initializer.

    Parameters are sampled from a zero-mean normal distribution with
    standard deviation

        sqrt(2 / fan_in)

    where ``fan_in`` is the effective number of input units contributing
    to each output.

    This initialization strategy helps preserve activation variance in
    deep feed-forward networks and is commonly used with ReLU-like
    activation functions.
    """

    def initialize(
        self,
        shape: tuple[int, ...],
        fan_in: int,
        fan_out: int,
    ) -> np.ndarray:
        """
        Generate He normal initialized parameters.

        Parameters
        ----------
        shape : tuple[int, ...]
            Desired parameter shape.

        fan_in : int
            Effective number of input units.

        fan_out : int
            Effective number of output units.

            Included for interface consistency but not used by He
            initialization.

        Returns
        -------
        np.ndarray
            Initialized parameter array.
        """

        return (
            np.random.randn(*shape)
            * np.sqrt(2.0 / fan_in)
        )


class HeUniform(Initializer):
    """
    He uniform parameter initializer.

    Parameters are sampled from a uniform distribution over the interval

        [-limit, limit]

    where

        limit = sqrt(6 / fan_in)

    and ``fan_in`` is the effective number of input units contributing
    to each output.

    This initialization strategy helps preserve activation variance in
    deep feed-forward networks and is commonly used with ReLU-like
    activation functions.
    """

    def initialize(
        self,
        shape: tuple[int, ...],
        fan_in: int,
        fan_out: int,
    ) -> np.ndarray:
        """
        Generate He uniform initialized parameters.

        Parameters
        ----------
        shape : tuple[int, ...]
            Desired parameter shape.

        fan_in : int
            Effective number of input units.

        fan_out : int
            Effective number of output units.

            Included for interface consistency but not used by He
            initialization.

        Returns
        -------
        np.ndarray
            Initialized parameter array.
        """

        limit = np.sqrt(6.0 / fan_in)

        return np.random.uniform(
            low=-limit,
            high=limit,
            size=shape,
        )