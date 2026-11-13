import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from modules.protocols import ExtractFeaturesFromDetectionResult
from modules.features.feature_transformer import FeatureTransformer

def load_dataset(root: Path,
                 feature_extracter: ExtractFeaturesFromDetectionResult, 
                 feature_transformer: FeatureTransformer = FeatureTransformer([]),
                 expected_no_of_frames: int = 30) -> tuple[np.ndarray, np.ndarray, list[str]]:

    sequences = []
    labels = []
    actions = []

    # First, enumerate through the directories that represent actions
    for action_index, action_path in enumerate(sorted(root.iterdir())):

        # If the action directory is prepended with an underscore, ignore it
        if not action_path.is_dir() or action_path.name.startswith('_'):
            continue
        
        action_index = len(actions)
        actions.append(action_path.name)

        sequence_paths = sorted(
            p for p in action_path.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        )

        for sequence_path in sequence_paths:
            frame_paths = sorted(sequence_path.glob("*.npy"))

            # Check if the number of frames matches the expected length
            if len(frame_paths) != expected_no_of_frames:
                raise ValueError(
                    f"{sequence_path}: expected {expected_no_of_frames} frames, "
                    f"got {len(frame_paths)}"
                )

            sequence = []
            for frame_path in frame_paths:
                # load the features from the file and transform it as data preprocessing
                feature = feature_extracter.load_feature(frame_path)
                feature = feature_transformer.transform(feature)
                
                sequence.append(feature.get_data())

            sequences.append(sequence)
            labels.append(action_index)

    print(f"Path {root} contains {len(actions)} actions and {len(sequences)} sequences.")
    
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(labels, dtype=np.int16),
        actions
    )

def save_model(model, model_file: Path) -> None:
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_file)