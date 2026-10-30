import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modules.protocols import Feature

class FeatureTransformer:
    def __init__(self, 
                 left_hand: float,
                 right_hand: float,
                 pose: float,
                 face: float):
        self._left_hand = left_hand
        self._right_hand = right_hand
        self._pose = pose
        self._face = face

    def transform(self, feature: Feature) -> Feature:
        return feature.apply_weights(left_hand=self._left_hand,
                                     right_hand=self._right_hand,
                                     pose=self._pose,
                                     face=self._face)