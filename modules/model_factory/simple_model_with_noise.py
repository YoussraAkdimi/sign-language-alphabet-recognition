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
from tensorflow.keras.regularizers import l2  # type: ignore

class SimpleModelWithNoise(ModelFactory):
    @property
    def name(self) -> str:
        return "simple_model_with_noise"

    @property
    def nr_of_epochs(self) -> int:
        return 1000
    
    def create(self, input_shape: tuple, action_length: int):
        model = Sequential([
            Input(shape=input_shape),
            GaussianNoise(0.01),
            GRU(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.1),
            GRU(32, dropout=0.3),
            Dense(32, activation="relu", kernel_regularizer=l2(1e-4)),
            Dropout(0.3),
            Dense(action_length, activation="softmax"),
        ])

        return model