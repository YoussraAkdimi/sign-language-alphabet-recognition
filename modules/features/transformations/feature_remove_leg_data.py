import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modules.protocols import Feature

class FeatureRemoveLegData:
    """
    More harmless than it sounds: remove lower body coordinates
    """
    def __init__(self):
        pass
    
    def transform(self, feature: Feature) -> Feature:
        """
        Apply transformation to the current feature and return as new feature.
        """
        return feature.remove_leg_data()