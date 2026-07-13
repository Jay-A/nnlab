import numpy as np
import pytest

from nnlab.initializers import (
    Initializer,
    HeNormal,
    HeUniform,
    XavierNormal,
    XavierUniform,
)


def test_initializer_base_is_abstract():
    """
    Verify base initializer cannot be instantiated.
    """

    with pytest.raises(TypeError):
        Initializer()


def test_he_normal_exists():
    """
    Verify He normal initializer constructs.
    """

    initializer = HeNormal()

    assert initializer is not None


def test_he_uniform_exists():
    """
    Verify He uniform initializer constructs.
    """

    initializer = HeUniform()

    assert initializer is not None


def test_xavier_normal_exists():
    """
    Verify Xavier normal initializer constructs.
    """

    initializer = XavierNormal()

    assert initializer is not None


def test_xavier_uniform_exists():
    """
    Verify Xavier uniform initializer constructs.
    """

    initializer = XavierUniform()

    assert initializer is not None


def test_he_normal_shape():
    """
    Verify He normal returns requested shape.
    """

    initializer = HeNormal()

    values = initializer.initialize(
        shape=(10, 5),
        fan_in=10,
        fan_out=5,
    )

    assert values.shape == (10, 5)


def test_he_uniform_shape():
    """
    Verify He uniform returns requested shape.
    """

    initializer = HeUniform()

    values = initializer.initialize(
        shape=(10, 5),
        fan_in=10,
        fan_out=5,
    )

    assert values.shape == (10, 5)


def test_xavier_normal_shape():
    """
    Verify Xavier normal returns requested shape.
    """

    initializer = XavierNormal()

    values = initializer.initialize(
        shape=(10, 5),
        fan_in=10,
        fan_out=5,
    )

    assert values.shape == (10, 5)


def test_xavier_uniform_shape():
    """
    Verify Xavier uniform returns requested shape.
    """

    initializer = XavierUniform()

    values = initializer.initialize(
        shape=(10, 5),
        fan_in=10,
        fan_out=5,
    )

    assert values.shape == (10, 5)


@pytest.mark.parametrize(
    "initializer",
    [
        HeNormal(),
        HeUniform(),
        XavierNormal(),
        XavierUniform(),
    ],
)
def test_initializer_values_are_finite(initializer):
    """
    Verify initializers produce finite values.
    """

    values = initializer.initialize(
        shape=(100, 50),
        fan_in=100,
        fan_out=50,
    )

    assert np.all(
        np.isfinite(values)
    )


def test_he_normal_variance():
    """
    Verify He normal approximately follows expected variance.
    """

    initializer = HeNormal()

    values = initializer.initialize(
        shape=(1000, 100),
        fan_in=1000,
        fan_out=100,
    )

    expected_variance = 2.0 / 1000

    assert np.isclose(
        np.var(values),
        expected_variance,
        rtol=0.2,
    )


def test_he_uniform_bounds():
    """
    Verify He uniform respects expected limits.
    """

    initializer = HeUniform()

    values = initializer.initialize(
        shape=(1000, 100),
        fan_in=1000,
        fan_out=100,
    )

    limit = np.sqrt(
        6.0 / 1000
    )

    assert np.max(values) <= limit
    assert np.min(values) >= -limit


def test_xavier_normal_variance():
    """
    Verify Xavier normal approximately follows expected variance.
    """

    initializer = XavierNormal()

    values = initializer.initialize(
        shape=(1000, 100),
        fan_in=1000,
        fan_out=100,
    )

    expected_variance = (
        2.0 / (1000 + 100)
    )

    assert np.isclose(
        np.var(values),
        expected_variance,
        rtol=0.2,
    )


def test_xavier_uniform_bounds():
    """
    Verify Xavier uniform respects expected limits.
    """

    initializer = XavierUniform()

    values = initializer.initialize(
        shape=(1000, 100),
        fan_in=1000,
        fan_out=100,
    )

    limit = np.sqrt(
        6.0 / (1000 + 100)
    )

    assert np.max(values) <= limit
    assert np.min(values) >= -limit