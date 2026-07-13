import numpy as np

from .base import FeatureExtractor


class SpectrogramExtractor(FeatureExtractor):
    """
    Extract magnitude spectrogram features from audio signals.

    A spectrogram represents the frequency content of an audio signal
    over time. The input signal is divided into overlapping windows,
    transformed into the frequency domain using the Fourier transform,
    and converted into magnitude values.

    The transformation pipeline is:

        audio waveform
            |
            v
        windowed frames
            |
            v
        FFT
            |
            v
        magnitude spectrum
            |
            v
        spectrogram

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.

    window_size : int, default=512
        Number of samples in each analysis window.

    hop_size : int, default=256
        Number of samples between consecutive windows.

    Notes
    -----
    The output representation has shape:

        (frequency_bins, time_frames)

    where:

        frequency_bins = window_size // 2 + 1

    This extractor performs deterministic feature transformation and
    does not contain trainable parameters.
    """

    def __init__(
        self,
        sample_rate: int,
        window_size: int = 512,
        hop_size: int = 256,
    ):
        """
        Initialize spectrogram extractor.

        Parameters
        ----------
        sample_rate : int
            Audio sample rate in Hz.

        window_size : int, default=512
            Number of samples per FFT window.

        hop_size : int, default=256
            Step size between consecutive windows.

        Raises
        ------
        ValueError
            If window_size or hop_size are not positive.
        """

        if window_size <= 0:
            raise ValueError(
                "window_size must be positive."
            )

        if hop_size <= 0:
            raise ValueError(
                "hop_size must be positive."
            )

        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size

        self.window = np.hanning(
            window_size,
        )

        self.last_output_shape = None

    @property
    def output_shape(
        self,
    ) -> tuple[int, ...] | None:
        """
        Return the most recently generated feature shape.

        Returns
        -------
        tuple[int, ...] | None
            Spectrogram shape:

                (frequency_bins, time_frames)

            Returns None before transform has been called.
        """

        return self.last_output_shape

    def transform(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Transform waveform into magnitude spectrogram.

        Parameters
        ----------
        x : np.ndarray
            One-dimensional audio waveform.

        Returns
        -------
        np.ndarray
            Magnitude spectrogram with shape:

                (frequency_bins, time_frames)

        Raises
        ------
        ValueError
            If input is not one-dimensional or shorter than one window.
        """

        if x.ndim != 1:
            raise ValueError(
                "Audio input must be a one-dimensional array."
            )

        if len(x) < self.window_size:
            raise ValueError(
                "Audio input must contain at least one window."
            )

        frames = []

        for start in range(
            0,
            len(x) - self.window_size + 1,
            self.hop_size,
        ):
            frame = x[
                start:start + self.window_size
            ]

            windowed = (
                frame
                * self.window
            )

            spectrum = np.fft.rfft(
                windowed,
            )

            magnitude = np.abs(
                spectrum,
            )

            frames.append(
                magnitude,
            )

        spectrogram = np.stack(
            frames,
            axis=1,
        )

        self.last_output_shape = spectrogram.shape

        return spectrogram