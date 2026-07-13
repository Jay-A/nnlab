import numpy as np

from nnlab.activations import ParameterizedActivation
from nnlab.kernels import LogisticKernel
from nnlab.layers import ActivationLayer, Dense
from nnlab.losses import MeanSquaredError
from nnlab.models import FeedForward
from nnlab.optimizers import SGD
from nnlab.training import Trainer


def test_trainer_exists():
    """
    Verify trainer constructs.
    """

    model = FeedForward(
        layers=[],
    )

    trainer = Trainer(
        model=model,
        loss=MeanSquaredError(),
        optimizer=SGD(),
    )

    assert trainer is not None


def test_trainer_predict_matches_model():
    """
    Verify trainer prediction delegates to the model.
    """

    model = FeedForward(
        layers=[],
    )

    trainer = Trainer(
        model=model,
        loss=MeanSquaredError(),
        optimizer=SGD(),
    )

    x = np.array(
        [
            [1.0],
            [2.0],
        ]
    )

    assert np.allclose(
        trainer.predict(x),
        model.forward(x),
    )


def test_trainer_fit_returns_history():
    """
    Verify fit returns one loss value per epoch.
    """

    model = FeedForward(
        layers=[
            Dense(
                input_size=1,
                output_size=1,
            ),
        ]
    )

    loss = MeanSquaredError()

    trainer = Trainer(
        model=model,
        loss=loss,
        optimizer=SGD(
            learning_rate=0.01,
        ),
    )

    x = np.array(
        [
            [0.0],
            [1.0],
        ]
    )

    target = np.array(
        [
            [0.0],
            [1.0],
        ]
    )

    history = trainer.fit(
        x,
        target,
        epochs=5,
    )

    assert len(history) == 5


def test_trainer_updates_model_parameters():
    """
    Verify training updates model parameters.
    """

    dense = Dense(
        input_size=1,
        output_size=1,
    )

    initial_weights = dense.weights.copy()

    model = FeedForward(
        layers=[
            dense,
        ]
    )

    trainer = Trainer(
        model=model,
        loss=MeanSquaredError(),
        optimizer=SGD(
            learning_rate=0.01,
        ),
    )

    x = np.array(
        [
            [0.0],
            [1.0],
        ]
    )

    target = np.array(
        [
            [0.0],
            [1.0],
        ]
    )

    trainer.fit(
        x,
        target,
        epochs=5,
    )

    assert not np.allclose(
        dense.weights,
        initial_weights,
    )


def test_trainer_reduces_loss():
    """
    Verify training reduces prediction error.
    """

    model = FeedForward(
        layers=[
            Dense(
                input_size=1,
                output_size=1,
            ),
            ActivationLayer(
                activation=ParameterizedActivation(
                    kernel=LogisticKernel(),
                )
            ),
        ]
    )

    trainer = Trainer(
        model=model,
        loss=MeanSquaredError(),
        optimizer=SGD(
            learning_rate=0.1,
        ),
    )

    x = np.linspace(
        -1.0,
        1.0,
        25,
    ).reshape(
        -1,
        1,
    )

    target = 1.0 / (
        1.0 + np.exp(-x)
    )

    history = trainer.fit(
        x,
        target,
        epochs=50,
    )

    assert history[-1] < history[0]


