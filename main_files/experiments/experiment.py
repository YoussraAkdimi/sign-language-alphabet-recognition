import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.features import HolisticLandmarksFeatureExtracter, FeatureTransformer
from modules.image_annotators import RealLifeTestAnnotator
from modules.model_factory import (SimpleModel,
                                  SimpleModelWithNoise)
from modules.features.transformations import FeatureWeights

from modules.process import (record_training_data, 
                            show_trained_data, 
                            train_and_validate_model,
                            real_life_test)
from modules.prediction_collectors import EMAPredictionCollector

from tensorflow.keras.models import load_model # type: ignore

DATA_PATH: Path = Path('./mp_data_test')
TENSORFLOW_MODEL_PATH: Path = Path('./output_models/simple_model.h5')
ACTIONS = ["test"]

feature_extracter = HolisticLandmarksFeatureExtracter()
feature_transformer = FeatureTransformer([FeatureWeights(0, 0, 0, 1.0)])

#record_training_data(DATA_PATH, ACTIONS, feature_extracter=feature_extracter)

# show trained data
show_trained_data(DATA_PATH, 
                  feature_extracter=feature_extracter, 
                  feature_transformer=feature_transformer)

exit()

# train_model
model = train_and_validate_model(model_factory=SimpleModelWithNoise(), 
                                 model_file=TENSORFLOW_MODEL_PATH,
                                 feature_extracter=feature_extracter,
                                 feature_transformer=feature_transformer,
                                 nr_of_epochs=50)

exit()

# load model
model = load_model(TENSORFLOW_MODEL_PATH)
gesture_prediction_collector = EMAPredictionCollector(threshold=0.4,
                                                      alpha=0.7,
                                                      commit_frames=10,
                                                      idle_frames_to_rearm=5,
                                                      idle_index=0)

real_life_test(trained_model=model, 
               gesture_prediction_collector=gesture_prediction_collector,
               image_annotator=RealLifeTestAnnotator())
