import numpy as np
import pytest

from nnlab.features import (
    FeatureExtractor,
    SpectrogramExtractor,
)


def test_feature_extractor_base_is_abstract():
    """
    Verify feature extractor base cannot be instantiated.
    """

    with pytest.raises(TypeError):
        FeatureExtractor()


def test_feature_extractor_default_output_shape():
    """
    Verify default output shape is unknown.
    """

    class DummyFeatureExtractor(FeatureExtractor):

        def transform(
            self,
            x: np.ndarray,
        ) -> np.ndarray:
            return x

    extractor = DummyFeatureExtractor()

    assert extractor.output_shape is None


def test_feature_extractor_transform():
    """
    Verify feature extractor transforms input data.
    """

    class DummyFeatureExtractor(FeatureExtractor):

        def transform(
            self,
            x: np.ndarray,
        ) -> np.ndarray:
            return x * 2.0

    extractor = DummyFeatureExtractor()

    x = np.array(
        [1.0, 2.0, 3.0],
    )

    result = extractor.transform(x)

    assert np.allclose(
        result,
        np.array(
            [2.0, 4.0, 6.0],
        ),
    )


def test_spectrogram_extractor_exists():
    """
    Verify spectrogram extractor constructs.
    """

    extractor = SpectrogramExtractor(
        sample_rate=16000,
    )

    assert extractor is not None


def test_spectrogram_output_shape():
    """
    Verify spectrogram output has expected dimensions.
    """

    extractor = SpectrogramExtractor(
        sample_rate=16000,
        window_size=256,
        hop_size=128,
    )

    x = np.zeros(
        1024,
    )

    result = extractor.transform(
        x,
    )

    assert result.shape == (
        129,
        7,
    )

    assert extractor.output_shape == result.shape


def test_spectrogram_detects_frequency_content():
    """
    Verify spectrogram captures frequency energy.

    A pure sine wave should produce a dominant frequency bin.
    """

    sample_rate = 16000
    frequency = 1000

    extractor = SpectrogramExtractor(
        sample_rate=sample_rate,
        window_size=512,
        hop_size=256,
    )

    time = np.arange(
        sample_rate,
    ) / sample_rate

    signal = np.sin(
        2.0
        * np.pi
        * frequency
        * time
    )

    spectrogram = extractor.transform(
        signal,
    )

    average_energy = np.mean(
        spectrogram,
        axis=1,
    )

    dominant_bin = np.argmax(
        average_energy,
    )

    expected_bin = int(
        frequency
        * extractor.window_size
        / sample_rate
    )

    assert abs(
        dominant_bin - expected_bin
    ) <= 1


def test_spectrogram_rejects_multidimensional_input():
    """
    Verify spectrogram requires one-dimensional audio.
    """

    extractor = SpectrogramExtractor(
        sample_rate=16000,
    )

    x = np.zeros(
        (2, 1000),
    )

    with pytest.raises(ValueError):
        extractor.transform(
            x,
        )


def test_spectrogram_rejects_short_input():
    """
    Verify spectrogram requires enough samples for a window.
    """

    extractor = SpectrogramExtractor(
        sample_rate=16000,
        window_size=512,
    )

    x = np.zeros(
        100,
    )

    with pytest.raises(ValueError):
        extractor.transform(
            x,
        )


def test_spectrogram_rejects_invalid_window_size():
    """
    Verify invalid window size raises an error.
    """

    with pytest.raises(ValueError):
        SpectrogramExtractor(
            sample_rate=16000,
            window_size=0,
        )


def test_spectrogram_rejects_invalid_hop_size():
    """
    Verify invalid hop size raises an error.
    """

    with pytest.raises(ValueError):
        SpectrogramExtractor(
            sample_rate=16000,
            hop_size=0,
        )