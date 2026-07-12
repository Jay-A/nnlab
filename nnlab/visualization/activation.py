import numpy as np
import matplotlib.pyplot as plt

from nnlab.activations import Activation


def plot_activation(
    activation: Activation,
    x_range: tuple[float, float] = (-5.0, 5.0),
    samples: int = 500,
    title: str | None = None,
):
    """
    Plot a single activation function.

    Parameters
    ----------
    activation : Activation
        Activation object to visualize.

    x_range : tuple[float, float]
        Minimum and maximum x values.

    samples : int
        Number of samples.

    title : str, optional
        Plot title.
    """

    x = np.linspace(
        x_range[0],
        x_range[1],
        samples,
    )

    y = activation.forward(x)

    plt.figure(
        figsize=(8, 4),
    )

    plt.plot(
        x,
        y,
        label=activation.__class__.__name__,
    )

    plt.axhline(
        0.0,
        color="black",
        linewidth=0.8,
    )

    plt.axvline(
        0.0,
        color="black",
        linewidth=0.8,
    )

    plt.grid(True)

    plt.xlabel("Input")
    plt.ylabel("Activation")

    if title:
        plt.title(title)

    plt.legend()

    plt.show()


def plot_activation_comparison(
    activations: dict[str, Activation],
    x_range: tuple[float, float] = (-5.0, 5.0),
    samples: int = 500,
):
    """
    Compare multiple activation functions.

    Parameters
    ----------
    activations : dict[str, Activation]
        Named activation functions.

    x_range : tuple[float, float]
        Input range.

    samples : int
        Number of samples.
    """

    x = np.linspace(
        x_range[0],
        x_range[1],
        samples,
    )

    plt.figure(
        figsize=(8, 5),
    )

    for name, activation in activations.items():

        y = activation.forward(x)

        plt.plot(
            x,
            y,
            label=name,
        )

    plt.axhline(
        0.0,
        color="black",
        linewidth=0.8,
    )

    plt.axvline(
        0.0,
        color="black",
        linewidth=0.8,
    )

    plt.grid(True)

    plt.xlabel("Input")
    plt.ylabel("Activation")

    plt.title(
        "Activation Function Comparison",
    )

    plt.legend()

    plt.show()


def plot_activation_derivative(
    activation: Activation,
    x_range: tuple[float, float] = (-5.0, 5.0),
    samples: int = 500,
):
    """
    Plot activation and derivative.

    Parameters
    ----------
    activation : Activation
        Activation object.

    x_range : tuple[float, float]
        Input range.

    samples : int
        Number of samples.
    """

    x = np.linspace(
        x_range[0],
        x_range[1],
        samples,
    )

    y = activation.forward(x)
    dy = activation.derivative(x)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
    )

    axes[0].plot(
        x,
        y,
    )

    axes[0].set_ylabel(
        "Activation",
    )

    axes[0].grid(True)

    axes[1].plot(
        x,
        dy,
    )

    axes[1].set_xlabel(
        "Input",
    )

    axes[1].set_ylabel(
        "Derivative",
    )

    axes[1].grid(True)

    fig.suptitle(
        activation.__class__.__name__,
    )

    plt.tight_layout()

    plt.show()