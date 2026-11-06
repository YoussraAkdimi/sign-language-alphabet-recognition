import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.features import HolisticLandmarksFeatureExtracter, FeatureTransformer
from modules.image_annotators import RealLifeTestAnnotator
from modules.model_factory import (SimpleModel,
                                   SimpleModelWithNoise,
                                   SimpleLSTMModel,
                                   SimpleLSTMModelWithNoise)

from modules.features.transformations import (FeatureWeights, 
                                              FeatureCentering,
                                              FeatureNormalizer,
                                              FeatureRemoveLegData,
                                              FeatureRemoveZCoordinates)

from modules.process import (record_training_data, 
                            show_trained_data, 
                            train_and_validate_model,
                            real_life_test)
from modules.prediction_collectors import EMAPredictionCollector
from modules.utilities.file import load_dataset, save_model
from tensorflow.keras.models import load_model # type: ignore

DATA_PATH: Path = Path('./mp_data')
DATA_PATH_VALIDATION: Path = Path('./mp_data_validation')
DATA_PATH_TEST: Path = Path('./mp_data_new_test_different_idle')
TENSORFLOW_MODEL_PATH: Path = Path('./output_models/simple_model.keras')
ACTIONS = ["idle", "poes", "slapen", "vogel"]

feature_extracter = HolisticLandmarksFeatureExtracter()
#feature_transformer = FeatureTransformer([FeatureCentering(), FeatureWeights(5.0, 5.0, 1.0, 0.5)])
feature_transformer = FeatureTransformer([FeatureCentering(), FeatureNormalizer(), FeatureWeights(10.0, 10.0, 2.0, 0.5)])
feature_transformer = FeatureTransformer([FeatureRemoveLegData(), FeatureCentering(), FeatureWeights(10.0, 10.0, 1.0, 0.5)])
feature_transformer = FeatureTransformer([FeatureRemoveZCoordinates()])


#record_training_data(DATA_PATH_VALIDATION, ACTIONS, feature_extracter=feature_extracter)

x_train, x_train_labels, actions = load_dataset(DATA_PATH,
                                feature_extracter=feature_extracter,
                                feature_transformer=feature_transformer,
                                expected_no_of_frames=30)

print(x_train[1:5])

x_validation, x_validation_labels, actions_validation = load_dataset(DATA_PATH_VALIDATION,
                                                feature_extracter=feature_extracter,
                                                feature_transformer=feature_transformer,
                                                expected_no_of_frames=30)

assert actions == actions_validation

exit()

# show trained data
show_trained_data(DATA_PATH, 
                 feature_extracter=feature_extracter, 
                 feature_transformer=feature_transformer)

exit()

# train_model
model, history = train_and_validate_model(model_factory=SimpleLSTMModelWithNoise(),
                         actions=actions,
                         train_dataset=x_train,
                         train_labels=x_train_labels,
                         validation_dataset=x_validation,
                         validation_labels=x_validation_labels,
                         nr_of_epochs=50)

save_model(model, TENSORFLOW_MODEL_PATH)

# load model
model = load_model(TENSORFLOW_MODEL_PATH)
gesture_prediction_collector = EMAPredictionCollector(threshold=0.7,
                                                      alpha=0.6,
                                                      commit_frames=10,
                                                      idle_frames_to_rearm=5,
                                                      idle_index=0)

real_life_test(trained_model=model, 
               feature_extracter=feature_extracter,
               feature_transformer=feature_transformer,
               gesture_prediction_collector=gesture_prediction_collector,
               image_annotator=RealLifeTestAnnotator())
