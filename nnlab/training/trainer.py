import numpy as np

from ..losses import Loss
from ..models import Model
from ..optimizers import Optimizer


class Trainer:
    """
    Coordinate neural network training.

    The trainer performs the complete optimization cycle for a model:

        forward
            -> loss
            -> backward
            -> optimizer update

    The trainer does not implement model behavior, loss functions,
    or optimization algorithms itself. Instead, it coordinates the
    interaction between these independent components.

    Parameters
    ----------
    model : Model
        Model to train.

    loss : Loss
        Loss function used to evaluate prediction error.

    optimizer : Optimizer
        Optimizer used to update model parameters.
    """

    def __init__(
        self,
        model: Model,
        loss: Loss,
        optimizer: Optimizer,
    ):
        """
        Initialize the trainer.

        Parameters
        ----------
        model : Model
            Model to optimize.

        loss : Loss
            Loss function.

        optimizer : Optimizer
            Parameter optimization algorithm.
        """

        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def fit(
        self,
        x: np.ndarray,
        target: np.ndarray,
        epochs: int = 1000,
    ) -> list[float]:
        """
        Train the model.

        Performs repeated optimization over the supplied training
        data for a fixed number of epochs.

        Parameters
        ----------
        x : np.ndarray
            Training inputs.

        target : np.ndarray
            Target outputs.

        epochs : int, default=1000
            Number of optimization iterations.

        Returns
        -------
        list[float]
            Loss value recorded after each epoch.
        """

        history = []

        for _ in range(epochs):

            prediction = self.model.forward(
                x,
            )

            error = self.loss.forward(
                prediction,
                target,
            )

            gradient = self.loss.derivative(
                prediction,
                target,
            )

            self.model.backward(
                gradient,
            )

            self.optimizer.step(
                self.model.parameters(),
                self.model.gradients(),
            )

            history.append(error)

        return history

    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Compute model predictions.

        Parameters
        ----------
        x : np.ndarray
            Input samples.

        Returns
        -------
        np.ndarray
            Model predictions.
        """

        return self.model.forward(
            x,
        )