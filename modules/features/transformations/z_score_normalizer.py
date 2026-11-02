import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modules.protocols import Feature
class FeatureNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self._mean = mean
        self._std = std

    def transform(self, feature: Feature) -> Feature:
        return feature.standardize(
            mean=self._mean,
            std=self._std
        )