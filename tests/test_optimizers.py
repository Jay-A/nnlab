import numpy as np
import pytest

from nnlab.optimizers import (
    Optimizer,
    SGD,
)


def test_optimizer_base_exists():
    """
    Verify optimizer base class is available.
    """

    assert Optimizer is not None


def test_optimizer_base_is_abstract():
    """
    Verify optimizer base cannot be instantiated directly.
    """

    with pytest.raises(TypeError):
        Optimizer()


def test_optimizer_requires_interface():
    """
    Verify child classes must implement optimizer methods.
    """

    class IncompleteOptimizer(Optimizer):
        pass

    with pytest.raises(TypeError):
        IncompleteOptimizer()


def test_sgd_exists():
    """
    Verify SGD optimizer constructs.
    """

    optimizer = SGD()

    assert optimizer is not None


def test_sgd_rejects_invalid_learning_rate():
    """
    Verify learning rate must be positive.
    """

    with pytest.raises(ValueError):
        SGD(
            learning_rate=0.0,
        )


def test_sgd_updates_parameters():
    """
    Verify SGD applies gradient descent update.
    """

    optimizer = SGD(
        learning_rate=0.1,
    )

    parameters = [
        np.array(
            [1.0, 2.0],
        ),
    ]

    gradients = [
        np.array(
            [0.5, -1.0],
        ),
    ]

    optimizer.step(
        parameters,
        gradients,
    )

    assert np.allclose(
        parameters[0],
        np.array(
            [
                0.95,
                2.10,
            ]
        ),
    )


def test_sgd_updates_multiple_parameters():
    """
    Verify SGD updates all supplied parameters.
    """

    optimizer = SGD(
        learning_rate=0.5,
    )

    parameters = [
        np.array(
            [1.0],
        ),
        np.array(
            [2.0],
        ),
    ]

    gradients = [
        np.array(
            [0.2],
        ),
        np.array(
            [-0.4],
        ),
    ]

    optimizer.step(
        parameters,
        gradients,
    )

    assert np.allclose(
        parameters[0],
        np.array([0.9]),
    )

    assert np.allclose(
        parameters[1],
        np.array([2.2]),
    )