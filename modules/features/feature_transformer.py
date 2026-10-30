import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.protocols import FeatureTransform, Feature

class FeatureTransformer:
    def __init__(self, transformations: list[FeatureTransform] | None = None):
        self._transformations = transformations or []

    def add(self, transformation: FeatureTransform) -> None:
        self._transformations.append(transformation)

    def transform(self, feature: Feature) -> Feature:
        #transformed_data = feature.get_data().copy()

        for transformation in self._transformations:
            feature = transformation.transform(feature)

        return feature
    
        #return type(feature)(transformed_data)