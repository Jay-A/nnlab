import numpy as np

from .base import Initializer


class XavierNormal(Initializer):
    """
    Xavier normal parameter initializer.

    Parameters are sampled from a zero-mean normal distribution with
    standard deviation

        sqrt(2 / (fan_in + fan_out))

    where ``fan_in`` is the effective number of input units and
    ``fan_out`` is the effective number of output units.

    Xavier initialization is designed to maintain activation variance
    during forward and backward propagation and is commonly used with
    activation functions such as sigmoid and tanh.
    """

    def initialize(
        self,
        shape: tuple[int, ...],
        fan_in: int,
        fan_out: int,
    ) -> np.ndarray:
        """
        Generate Xavier normal initialized parameters.

        Parameters
        ----------
        shape : tuple[int, ...]
            Desired parameter shape.

        fan_in : int
            Effective number of input units.

        fan_out : int
            Effective number of output units.

        Returns
        -------
        np.ndarray
            Initialized parameter array.
        """

        std = np.sqrt(
            2.0 / (fan_in + fan_out)
        )

        return (
            np.random.randn(*shape)
            * std
        )


class XavierUniform(Initializer):
    """
    Xavier uniform parameter initializer.

    Parameters are sampled from a uniform distribution over the interval

        [-limit, limit]

    where

        limit = sqrt(6 / (fan_in + fan_out))

    and ``fan_in`` and ``fan_out`` are the effective input and output
    dimensions.

    Xavier initialization is designed to maintain activation variance
    during forward and backward propagation and is commonly used with
    activation functions such as sigmoid and tanh.
    """

    def initialize(
        self,
        shape: tuple[int, ...],
        fan_in: int,
        fan_out: int,
    ) -> np.ndarray:
        """
        Generate Xavier uniform initialized parameters.

        Parameters
        ----------
        shape : tuple[int, ...]
            Desired parameter shape.

        fan_in : int
            Effective number of input units.

        fan_out : int
            Effective number of output units.

        Returns
        -------
        np.ndarray
            Initialized parameter array.
        """

        limit = np.sqrt(
            6.0 / (fan_in + fan_out)
        )

        return np.random.uniform(
            low=-limit,
            high=limit,
            size=shape,
        )