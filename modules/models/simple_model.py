from modules.protocols import ModelFactory
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dropout, Input, LSTM, Dense # type: ignore

class SimpleModel(ModelFactory):
    def create(self, input_shape: tuple, action_length: int):
        model = Sequential([
            Input(shape=input_shape),

            LSTM(32, return_sequences=True),
            LSTM(16),

            Dense(16, activation="relu"),
            Dense(action_length, activation="softmax")
        ])

        return model