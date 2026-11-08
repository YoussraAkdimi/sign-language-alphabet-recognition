import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.protocols import FeatureTransform, Feature

class FeatureTransformer:
    """
    This class hosts zero or more feature transformations and 
    applies them in sequence to a given Feature object.
    """
    def __init__(self, transformations: list[FeatureTransform] | None = None):
        self._transformations = transformations or []

    def add(self, transformation: FeatureTransform) -> None:
        """
        Add a new feature transformation to the list of transformations.
        """
        self._transformations.append(transformation)

    def transform(self, feature: Feature) -> Feature:
        """
        Apply all transformations in sequence to the given feature and return the transformed feature.
        """
        for transformation in self._transformations:
            feature = transformation.transform(feature)

        return feature