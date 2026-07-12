import numpy as np
import pytest

from nnlab.models import FeedForward, Model
from nnlab.layers import Dense


def test_model_base_is_abstract():
    """
    Verify model base cannot be instantiated.
    """

    with pytest.raises(TypeError):
        Model()


def test_feedforward_exists():
    """
    Verify feed-forward model constructs.
    """

    model = FeedForward(
        layers=[],
    )

    assert model is not None


def test_feedforward_applies_layers_in_order():
    """
    Verify model applies layers sequentially.
    """

    layer1 = Dense(
        input_size=2,
        output_size=2,
    )

    layer2 = Dense(
        input_size=2,
        output_size=1,
    )

    # Set deterministic transformations:
    #
    # layer1:
    # [x1, x2] -> [x1 + x2, x1 - x2]
    #
    layer1.weights = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
        ]
    )

    layer1.bias = np.array(
        [0.0, 0.0],
    )

    # layer2:
    # [a, b] -> [a + b]
    #
    layer2.weights = np.array(
        [
            [1.0],
            [1.0],
        ]
    )

    layer2.bias = np.array(
        [0.0],
    )

    model = FeedForward(
        layers=[
            layer1,
            layer2,
        ]
    )

    x = np.array(
        [
            [2.0, 3.0],
        ]
    )

    y = model.forward(x)

    # layer1:
    # [2,3] -> [5,-1]
    #
    # layer2:
    # [5,-1] -> [4]
    #
    assert np.allclose(
        y,
        np.array([[4.0]]),
    )


def test_feedforward_backward_propagates_gradients():
    """
    Verify model propagates gradients through layers in reverse order.
    """

    layer1 = Dense(
        input_size=2,
        output_size=2,
    )

    layer2 = Dense(
        input_size=2,
        output_size=1,
    )

    layer1.weights = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
        ]
    )

    layer1.bias = np.array(
        [0.0, 0.0],
    )

    layer2.weights = np.array(
        [
            [1.0],
            [1.0],
        ]
    )

    layer2.bias = np.array(
        [0.0],
    )

    model = FeedForward(
        layers=[
            layer1,
            layer2,
        ]
    )

    x = np.array(
        [
            [2.0, 3.0],
        ]
    )

    prediction = model.forward(
        x,
    )

    gradient = np.ones_like(
        prediction,
    )

    input_gradient = model.backward(
        gradient,
    )

    assert np.allclose(
        input_gradient,
        np.array(
            [
                [2.0, 0.0],
            ]
        ),
    )