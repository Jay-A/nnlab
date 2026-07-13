from abc import ABC, abstractmethod

import numpy as np


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extraction.

    Feature extractors transform raw input data into numerical
    representations suitable for machine learning models.

    Feature extraction is performed before model evaluation:

        raw input -> feature representation -> neural network

    Feature extractors do not perform learning or parameter updates.
    They provide deterministic transformations of input data.
    """

    @abstractmethod
    def transform(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Transform raw input data into features.

        Parameters
        ----------
        x : np.ndarray
            Raw input data.

        Returns
        -------
        np.ndarray
            Extracted feature representation.
        """
        pass

    @property
    def output_shape(
        self,
    ) -> tuple[int, ...] | None:
        """
        Shape of the feature representation produced.

        Returns
        -------
        tuple[int, ...] | None
            Expected output shape of transformed features.

            Returns None when the output shape depends on the
            input data or cannot be determined before transformation.
        """

        return None
        