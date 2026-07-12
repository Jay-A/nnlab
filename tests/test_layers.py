import numpy as np
import pytest

from nnlab.activations import ParameterizedActivation
from nnlab.kernels import LogisticKernel
from nnlab.layers import ActivationLayer, Dense, Layer


def test_layer_base_is_abstract():
    """
    Verify base layer cannot be instantiated.
    """

    with pytest.raises(TypeError):
        Layer()


def test_dense_exists():
    """
    Verify dense layer constructs.
    """

    layer = Dense(
        input_size=3,
        output_size=2,
    )

    assert layer is not None


def test_dense_forward_shape():
    """
    Verify dense layer output shape.
    """

    layer = Dense(
        input_size=3,
        output_size=2,
    )

    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    y = layer.forward(x)

    assert y.shape == (2, 2)


def test_dense_forward_values():
    """
    Verify dense layer performs affine transformation.
    """

    layer = Dense(
        input_size=2,
        output_size=1,
    )

    layer.weights = np.array(
        [
            [2.0],
            [3.0],
        ]
    )

    layer.bias = np.array(
        [1.0],
    )

    x = np.array(
        [
            [1.0, 1.0],
        ]
    )

    y = layer.forward(x)

    assert np.allclose(
        y,
        np.array([[6.0]]),
    )


def test_activation_layer_exists():
    """
    Verify activation layer constructs.
    """

    layer = ActivationLayer(
        activation=ParameterizedActivation(
            kernel=LogisticKernel(),
        )
    )

    assert layer is not None


def test_activation_layer_forward_shape():
    """
    Verify activation layer preserves input shape.
    """

    layer = ActivationLayer(
        activation=ParameterizedActivation(
            kernel=LogisticKernel(),
        )
    )

    x = np.linspace(
        -5.0,
        5.0,
        100,
    )

    y = layer.forward(x)

    assert y.shape == x.shape


def test_activation_layer_matches_activation():
    """
    Verify activation layer delegates computation.
    """

    activation = ParameterizedActivation(
        kernel=LogisticKernel(),
    )

    layer = ActivationLayer(
        activation=activation,
    )

    x = np.array(
        [
            -1.0,
            0.0,
            1.0,
        ]
    )

    assert np.allclose(
        layer.forward(x),
        activation.forward(x),
    )