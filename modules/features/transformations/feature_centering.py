import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modules.protocols import Feature
import numpy as np

class FeatureCentering:
    """
    A class that centers the features of a given Feature object based on pose landmarks.
    """
    def __init__(self):
        pass

    def transform(self, feature: Feature) -> Feature:
        """
        Apply centering to the current feature and return as new feature.
        """
        if feature.get_data() is None:
            return feature

        # delegate the operation to the feature itself
        return feature.center_at_pose()
    