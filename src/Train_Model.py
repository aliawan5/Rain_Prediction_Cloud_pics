import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import logging


class Trainer:
    def __init__(self, learning_rate, model, epochs, train_data) -> None:
        self.learning_rate = learning_rate
        self.model = model
        self.epochs = epochs
        self.train_data = train_data


    def compile_model(self):
        try:
            logging.info("Compiling the model....")
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

        except Exception as e:
            logging.error(f"Error occurred while compiling the model: {str(e)}")
            raise e
        

    def train_model(self):
        try:
            logging.info("Training model....")
            self.model.fit(self.train_data, epochs=self.epochs)

        except Exception as e:
            logging.error(f"Error occurred while training the model: {str(e)}")
            raise e
        