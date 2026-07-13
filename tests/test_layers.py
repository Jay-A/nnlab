import numpy as np
import pytest

from nnlab.activations import ParameterizedActivation
from nnlab.kernels import LogisticKernel
from nnlab.layers import ActivationLayer, Dense, Layer
from nnlab.initializers import Initializer


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


def test_dense_backward_shapes():
    """
    Verify dense backward pass preserves expected shapes.
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

    output = layer.forward(x)

    gradient = np.ones_like(output)

    input_gradient = layer.backward(
        gradient,
    )

    assert input_gradient.shape == x.shape

    assert layer.weight_gradient.shape == (
        3,
        2,
    )

    assert layer.bias_gradient.shape == (
        2,
    )


def test_dense_backward_values():
    """
    Verify dense backward computes parameter gradients.
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
            [1.0, 2.0],
        ]
    )

    output = layer.forward(x)

    gradient = np.array(
        [
            [1.0],
        ]
    )

    input_gradient = layer.backward(
        gradient,
    )

    assert np.allclose(
        layer.weight_gradient,
        np.array(
            [
                [1.0],
                [2.0],
            ]
        ),
    )

    assert np.allclose(
        layer.bias_gradient,
        np.array(
            [1.0],
        ),
    )

    assert np.allclose(
        input_gradient,
        np.array(
            [
                [2.0, 3.0],
            ]
        ),
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


def test_activation_layer_backward_shape():
    """
    Verify activation layer backward preserves shape.
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

    output = layer.forward(x)

    gradient = np.ones_like(output)

    input_gradient = layer.backward(
        gradient,
    )

    assert input_gradient.shape == x.shape
    assert np.all(np.isfinite(input_gradient))


def test_dense_parameters():
    """
    Verify dense layer exposes trainable parameters.
    """

    layer = Dense(
        input_size=3,
        output_size=2,
    )

    parameters = layer.parameters()

    assert len(parameters) == 2

    assert parameters[0] is layer.weights
    assert parameters[1] is layer.bias    


def test_activation_layer_has_no_parameters():
    """
    Verify activation layers expose no trainable parameters.
    """

    layer = ActivationLayer(
        activation=ParameterizedActivation(
            kernel=LogisticKernel(),
        )
    )

    parameters = layer.parameters()

    assert parameters == []


def test_activation_layer_backward_values():
    """
    Verify activation layer propagates gradients.
    """

    activation = ParameterizedActivation(
        kernel=LogisticKernel(),
    )

    layer = ActivationLayer(
        activation=activation,
    )

    x = np.array(
        [
            0.0,
        ]
    )

    layer.forward(x)

    gradient = np.array(
        [
            1.0,
        ]
    )

    result = layer.backward(
        gradient,
    )

    expected = activation.derivative(x)

    assert np.allclose(
        result,
        expected,
    )    


def test_dense_parameters():
    """
    Verify dense exposes trainable parameters.
    """

    layer = Dense(
        input_size=2,
        output_size=1,
    )

    parameters = layer.parameters()

    assert len(parameters) == 2
    assert parameters[0].shape == (2, 1)
    assert parameters[1].shape == (1,)


def test_activation_layer_has_no_parameters():
    """
    Verify activation layers are non-trainable.
    """

    layer = ActivationLayer(
        activation=ParameterizedActivation(
            kernel=LogisticKernel(),
        )
    )

    assert layer.parameters() == []
    assert layer.gradients() == []    


def test_dense_default_initializer():
    """
    Verify dense uses a default initializer when none is provided.
    """

    layer = Dense(
        input_size=4,
        output_size=3,
    )

    assert layer.weights.shape == (
        4,
        3,
    )

    assert np.all(np.isfinite(layer.weights))    


class ZeroInitializer(Initializer):
    """
    Test initializer that returns zeros.
    """

    def initialize(
        self,
        shape: tuple[int, ...],
        fan_in: int,
        fan_out: int,
    ) -> np.ndarray:

        return np.zeros(
            shape,
        )


def test_dense_custom_initializer():
    """
    Verify dense accepts custom initializers.
    """

    layer = Dense(
        input_size=2,
        output_size=3,
        initializer=ZeroInitializer(),
    )

    assert np.allclose(
        layer.weights,
        np.zeros(
            (2, 3),
        ),
    )    


class RecordingInitializer(Initializer):
    """
    Test initializer that records initialization metadata.
    """

    def __init__(self):
        self.shape = None
        self.fan_in = None
        self.fan_out = None

    def initialize(
        self,
        shape: tuple[int, ...],
        fan_in: int,
        fan_out: int,
    ) -> np.ndarray:

        self.shape = shape
        self.fan_in = fan_in
        self.fan_out = fan_out

        return np.zeros(shape)


def test_dense_passes_initializer_shape_information():
    """
    Verify dense passes correct shape information.
    """

    initializer = RecordingInitializer()

    Dense(
        input_size=5,
        output_size=7,
        initializer=initializer,
    )

    assert initializer.shape == (
        5,
        7,
    )

    assert initializer.fan_in == 5
    assert initializer.fan_out == 7    