import numpy as np
import pytest

from nnlab.kernels import (
    ArctangentKernel,
    GaussianCDFKernel,
    GaussianRBFKernel,
    LogisticKernel,
    PiecewiseLinearKernel,
    PolynomialKernel,
    TransitionKernel,
)


def test_kernel_base_exists():
    """
    Verify the kernel base class is available.
    """

    assert TransitionKernel is not None


def test_kernel_classes_exist():
    """
    Verify all V0.1 kernels instantiate correctly.
    """

    kernels = [
        LogisticKernel(),
        ArctangentKernel(),
        GaussianCDFKernel(),
        GaussianRBFKernel(),
        PiecewiseLinearKernel(),
        PolynomialKernel(),
    ]

    for kernel in kernels:
        assert isinstance(kernel, TransitionKernel)


def test_kernel_forward_shapes():
    """
    Verify kernels preserve input array shape.
    """

    x = np.linspace(-5.0, 5.0, 100)

    kernels = [
        LogisticKernel(),
        ArctangentKernel(),
        GaussianCDFKernel(),
        GaussianRBFKernel(),
        PiecewiseLinearKernel(),
        PolynomialKernel(),
    ]

    for kernel in kernels:
        y = kernel.forward(x)

        assert y.shape == x.shape
        assert np.all(np.isfinite(y))


def test_transition_kernel_ranges():
    """
    Verify transition kernels produce values in [0, 1].
    """

    x = np.linspace(-10.0, 10.0, 1000)

    kernels = [
        LogisticKernel(),
        ArctangentKernel(),
        GaussianCDFKernel(),
        PiecewiseLinearKernel(),
        PolynomialKernel(),
    ]

    for kernel in kernels:
        y = kernel.forward(x)

        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)


def test_gaussian_rbf_properties():
    """
    Verify Gaussian RBF has a maximum response at zero.
    """

    kernel = GaussianRBFKernel()

    x = np.array([-5.0, 0.0, 5.0])
    y = kernel.forward(x)

    assert y[1] == 1.0
    assert y[0] < y[1]
    assert y[2] < y[1]


def test_logistic_kernel_limits():
    """
    Verify logistic kernel approaches 0 and 1 at extreme inputs.
    """

    kernel = LogisticKernel()

    x = np.array([-100.0, 100.0])
    y = kernel.forward(x)

    assert y[0] < 1e-10
    assert y[1] > 1.0 - 1e-10


def test_kernel_derivatives():
    """
    Verify kernel derivatives return finite values.
    """

    x = np.linspace(-5.0, 5.0, 100)

    kernels = [
        LogisticKernel(),
        ArctangentKernel(),
        GaussianCDFKernel(),
        GaussianRBFKernel(),
        PiecewiseLinearKernel(),
        PolynomialKernel(),
    ]

    for kernel in kernels:
        dy = kernel.derivative(x)

        assert dy.shape == x.shape
        assert np.all(np.isfinite(dy))


def test_kernel_base_is_abstract():
    """
    Verify the base kernel cannot be instantiated directly.
    """

    with pytest.raises(TypeError):
        TransitionKernel()        