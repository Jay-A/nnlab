import numpy as np
import pytest

from nnlab.activations import (
    Activation,
    ParameterizedActivation,
)

from nnlab.kernels import LogisticKernel


def test_activation_base_exists():
    """
    Verify the activation base class is available.
    """

    assert Activation is not None


def test_activation_base_is_abstract():
    """
    Verify the activation base cannot be instantiated directly.
    """

    with pytest.raises(TypeError):
        Activation()


def test_activation_requires_interface():
    """
    Verify child classes must implement activation methods.
    """

    class IncompleteActivation(Activation):
        pass

    with pytest.raises(TypeError):
        IncompleteActivation()


def test_activation_example_implementation():
    """
    Verify a complete activation implementation works.
    """

    class TestActivation(Activation):

        def forward(self, x: np.ndarray) -> np.ndarray:
            return x

        def derivative(self, x: np.ndarray) -> np.ndarray:
            return np.ones_like(x)

    activation = TestActivation()

    x = np.array([1.0, 2.0, 3.0])

    assert np.all(activation.forward(x) == x)
    assert np.all(activation.derivative(x) == 1.0)


def test_parameterized_activation_exists():
    """
    Verify parameterized activation can be constructed.
    """

    activation = ParameterizedActivation(
        kernel=LogisticKernel()
    )

    assert activation is not None


def test_parameterized_activation_forward():
    """
    Verify parameterized activation evaluates kernel output.
    """

    activation = ParameterizedActivation(
        kernel=LogisticKernel()
    )

    x = np.array([-1.0, 0.0, 1.0])

    y = activation.forward(x)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_parameterized_activation_parameters():
    """
    Verify activation parameters modify the transformation.
    """

    x = np.array([0.0])

    normal = ParameterizedActivation(
        kernel=LogisticKernel()
    )

    shifted = ParameterizedActivation(
        kernel=LogisticKernel(),
        center=1.0,
    )

    assert not np.isclose(
        normal.forward(x),
        shifted.forward(x),
    )


def test_parameterized_activation_width():
    """
    Verify width changes activation sharpness.
    """

    x = np.array([1.0])

    narrow = ParameterizedActivation(
        kernel=LogisticKernel(),
        width=0.1,
    )

    wide = ParameterizedActivation(
        kernel=LogisticKernel(),
        width=10.0,
    )

    assert not np.isclose(
        narrow.forward(x),
        wide.forward(x),
    )


def test_parameterized_activation_derivative():
    """
    Verify activation derivative returns finite values.
    """

    activation = ParameterizedActivation(
        kernel=LogisticKernel()
    )

    x = np.linspace(-5.0, 5.0, 100)

    dy = activation.derivative(x)

    assert dy.shape == x.shape
    assert np.all(np.isfinite(dy))


def test_parameterized_activation_rejects_invalid_width():
    """
    Verify activation width must be positive.
    """

    with pytest.raises(ValueError):
        ParameterizedActivation(
            kernel=LogisticKernel(),
            width=0.0,
        )    


def test_parameterized_activation_scale_bias():
    """
    Verify scale and bias transform activation output.
    """

    x = np.array([0.0])

    base = ParameterizedActivation(
        kernel=LogisticKernel(),
    )

    transformed = ParameterizedActivation(
        kernel=LogisticKernel(),
        scale=2.0,
        bias=1.0,
    )

    assert not np.isclose(
        base.forward(x),
        transformed.forward(x),
    )