import numpy as np
import pytest

from nnlab.losses import (
    HuberLoss,
    LogCoshLoss,
    Loss,
    MeanAbsoluteError,
    MeanSquaredError,
)


def test_loss_base_exists():
    """
    Verify the loss base class is available.
    """

    assert Loss is not None


def test_loss_base_is_abstract():
    """
    Verify the base loss cannot be instantiated directly.
    """

    with pytest.raises(TypeError):
        Loss()


def test_loss_requires_interface():
    """
    Verify child classes must implement loss methods.
    """

    class IncompleteLoss(Loss):
        pass

    with pytest.raises(TypeError):
        IncompleteLoss()


def test_loss_example_implementation():
    """
    Verify a complete loss implementation works.
    """

    class TestLoss(Loss):

        def forward(
            self,
            prediction: np.ndarray,
            target: np.ndarray,
        ) -> float:
            return float(np.mean(prediction - target))

        def derivative(
            self,
            prediction: np.ndarray,
            target: np.ndarray,
        ) -> np.ndarray:
            return np.ones_like(prediction)

    loss = TestLoss()

    prediction = np.array([1.0, 2.0, 3.0])
    target = np.array([1.0, 1.0, 1.0])

    assert loss.forward(prediction, target) == 1.0
    assert np.all(
        loss.derivative(prediction, target) == 1.0
    )


def test_loss_classes_exist():
    """
    Verify all implemented losses construct correctly.
    """

    losses = [
        MeanSquaredError(),
        MeanAbsoluteError(),
        HuberLoss(),
        LogCoshLoss(),
    ]

    for loss in losses:
        assert isinstance(loss, Loss)


def test_loss_forward_returns_scalar():
    """
    Verify losses return scalar float values.
    """

    prediction = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    target = np.zeros_like(prediction)

    losses = [
        MeanSquaredError(),
        MeanAbsoluteError(),
        HuberLoss(),
        LogCoshLoss(),
    ]

    for loss in losses:
        value = loss.forward(
            prediction,
            target,
        )

        assert isinstance(value, float)
        assert np.isfinite(value)


def test_mse_forward_value():
    """
    Verify mean squared error computation.
    """

    loss = MeanSquaredError()

    prediction = np.array(
        [1.0, 2.0, 3.0]
    )

    target = np.array(
        [0.0, 0.0, 0.0]
    )

    value = loss.forward(
        prediction,
        target,
    )

    assert np.isclose(
        value,
        14.0 / 3.0,
    )


def test_mae_forward_value():
    """
    Verify mean absolute error computation.
    """

    loss = MeanAbsoluteError()

    prediction = np.array(
        [1.0, -2.0, 3.0]
    )

    target = np.zeros(3)

    value = loss.forward(
        prediction,
        target,
    )

    assert np.isclose(
        value,
        2.0,
    )


def test_huber_forward_quadratic_region():
    """
    Verify Huber loss uses quadratic region for
    small errors.
    """

    loss = HuberLoss(
        delta=2.0,
    )

    prediction = np.array(
        [1.0],
    )

    target = np.array(
        [0.0],
    )

    value = loss.forward(
        prediction,
        target,
    )

    assert np.isclose(
        value,
        0.5,
    )


def test_huber_forward_linear_region():
    """
    Verify Huber loss uses linear region for
    large errors.
    """

    loss = HuberLoss(
        delta=1.0,
    )

    prediction = np.array(
        [3.0],
    )

    target = np.array(
        [0.0],
    )

    value = loss.forward(
        prediction,
        target,
    )

    expected = 1.0 * (3.0 - 0.5)

    assert np.isclose(
        value,
        expected,
    )


def test_log_cosh_zero_error():
    """
    Verify log-cosh is zero when predictions match targets.
    """

    loss = LogCoshLoss()

    prediction = np.array(
        [1.0, 2.0, 3.0],
    )

    target = prediction.copy()

    value = loss.forward(
        prediction,
        target,
    )

    assert np.isclose(
        value,
        0.0,
    )


def test_loss_derivatives_shape():
    """
    Verify loss derivatives preserve prediction shape.
    """

    prediction = np.linspace(
        -5.0,
        5.0,
        100,
    )

    target = np.zeros_like(prediction)

    losses = [
        MeanSquaredError(),
        MeanAbsoluteError(),
        HuberLoss(),
        LogCoshLoss(),
    ]

    for loss in losses:
        gradient = loss.derivative(
            prediction,
            target,
        )

        assert gradient.shape == prediction.shape


def test_loss_derivatives_are_finite():
    """
    Verify loss derivatives return finite values.
    """

    prediction = np.linspace(
        -5.0,
        5.0,
        100,
    )

    target = np.zeros_like(prediction)

    losses = [
        MeanSquaredError(),
        MeanAbsoluteError(),
        HuberLoss(),
        LogCoshLoss(),
    ]

    for loss in losses:
        gradient = loss.derivative(
            prediction,
            target,
        )

        assert np.all(
            np.isfinite(gradient)
        )


def test_mse_derivative():
    """
    Verify MSE derivative matches analytical gradient.
    """

    loss = MeanSquaredError()

    prediction = np.array(
        [1.0, 2.0],
    )

    target = np.array(
        [0.0, 1.0],
    )

    gradient = loss.derivative(
        prediction,
        target,
    )

    expected = np.array(
        [1.0, 1.0],
    )

    assert np.allclose(
        gradient,
        expected,
    )


def test_mae_derivative():
    """
    Verify MAE subgradient.
    """

    loss = MeanAbsoluteError()

    prediction = np.array(
        [-2.0, 0.0, 2.0],
    )

    target = np.zeros(3)

    gradient = loss.derivative(
        prediction,
        target,
    )

    expected = np.array(
        [-1 / 3, 0.0, 1 / 3],
    )

    assert np.allclose(
        gradient,
        expected,
    )


def test_log_cosh_derivative_at_zero():
    """
    Verify log-cosh derivative is zero at zero error.
    """

    loss = LogCoshLoss()

    prediction = np.array(
        [0.0],
    )

    target = np.array(
        [0.0],
    )

    gradient = loss.derivative(
        prediction,
        target,
    )

    assert np.isclose(
        gradient[0],
        0.0,
    )


def test_huber_rejects_invalid_delta():
    """
    Verify Huber delta must be positive.
    """

    with pytest.raises(ValueError):
        HuberLoss(
            delta=0.0,
        )


def test_losses_are_zero_when_matching():
    """
    Verify all losses vanish when predictions equal targets.
    """

    prediction = np.array(
        [1.0, 2.0, 3.0],
    )

    target = prediction.copy()

    losses = [
        MeanSquaredError(),
        MeanAbsoluteError(),
        HuberLoss(),
        LogCoshLoss(),
    ]

    for loss in losses:
        assert np.isclose(
            loss.forward(
                prediction,
                target,
            ),
            0.0,
        )