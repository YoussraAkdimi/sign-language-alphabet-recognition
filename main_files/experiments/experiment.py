import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.image_annotators import RealLifeTestAnnotator
from modules.model_factory import SimpleModel
from modules.process import (record_training_data, 
                            show_trained_data, 
                            train_and_validate_model,
                            real_life_test)
from modules.prediction_collectors import EMAPredictionCollector

from tensorflow.keras.models import load_model # type: ignore

DATA_PATH: Path = Path('./mp_data')
TENSORFLOW_MODEL_PATH: Path = Path('./output_models/simple_model.h5')
ACTIONS = ["test"]

#record_training_data(DATA_PATH, ACTIONS)

# show trained data
#show_trained_data(DATA_PATH)

# train_model
#model = train_and_validate_model(model_factory=SimpleModel(), 
#                                 model_file=TENSORFLOW_MODEL_PATH,
#                                 nr_of_epochs=400)

# load model
model = load_model(TENSORFLOW_MODEL_PATH)
gesture_prediction_collector = EMAPredictionCollector(threshold=0.4,
                                                      alpha=0.6,
                                                      commit_frames=5,
                                                      idle_frames_to_rearm=5,
                                                      idle_index=0)

real_life_test(trained_model=model, 
               gesture_prediction_collector=gesture_prediction_collector,
               image_annotator=RealLifeTestAnnotator())
