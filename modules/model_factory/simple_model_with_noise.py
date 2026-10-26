from modules.protocols import ModelFactory
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import (
    Dropout, 
    Input, 
    LSTM, 
    Dense, 
    GRU,
    GaussianNoise
) # type: ignore

class SimpleModel(ModelFactory):
    @property
    def name(self) -> str:
        return "simple_model"

    @property
    def nr_of_epochs(self) -> int:
        return 1000
    
    def create(self, input_shape: tuple, action_length: int):
        model = Sequential([
            Input(shape=input_shape),
            GaussianNoise(0.01),          # only active during training
            GRU(64, return_sequences=True),
            GRU(32),
            Dense(32, activation="relu"),
            Dense(action_length, activation="softmax"),
        ])

        return model