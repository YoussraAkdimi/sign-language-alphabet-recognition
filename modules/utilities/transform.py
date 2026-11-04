import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from modules.protocols import ExtractFeaturesFromDetectionResult
from modules.features.feature_transformer import FeatureTransformer
from modules.protocols.protocols import Feature
from modules.features.holistic_landmarks_feature import HolisticLandmarksFeature

def transform_datasets(
    datasets: list[np.ndarray],
    feature_transformer: FeatureTransformer
) -> list[np.ndarray]:
    """
    Transforms multiple datasets using the provided feature transformer.

    Args:
        datasets: A list of datasets to transform.
        feature_transformer: The feature transformer to apply.

    Returns:
        A list containing the transformed datasets.
    """
    return [
        transform_dataset(dataset, feature_transformer)
        for dataset in datasets
    ]

def transform_dataset(dataset: np.ndarray, 
                      feature_transformer: FeatureTransformer) -> np.ndarray:
    """
    Transforms the dataset using the provided feature transformer.

    Args:
        dataset (np.ndarray): The dataset to be transformed.
        feature_transformer (FeatureTransformer): The feature transformer to apply.

    Returns:
        np.ndarray: The transformed dataset.
    """
    transformed_sequences = []
    
    for sequence in dataset:
        transformed_sequence = []
        for frame in sequence:
            feature = HolisticLandmarksFeature(data=frame)

            # Transform the feature using the feature transformer
            transformed_feature = feature_transformer.transform(feature)
            transformed_sequence.append(transformed_feature.get_data())
        
        transformed_sequences.append(transformed_sequence)
    
    return np.asarray(transformed_sequences, dtype=np.float32)