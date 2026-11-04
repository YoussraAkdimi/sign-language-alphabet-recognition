import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modules.protocols import Feature
import numpy as np

class FeatureHandNormalizer:
    def __init__(self):
        pass

    def transform(self, feature: Feature) -> Feature:
        if feature.get_data() is None:
            return feature

        return feature.normalize_hands()