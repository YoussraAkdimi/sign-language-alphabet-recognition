import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modules.protocols import Feature

class FeatureRemoveLegData:
    def __init__(self):
        pass
    
    def transform(self, feature: Feature) -> Feature:
        return feature.remove_leg_data()