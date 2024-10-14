import tensorflow as tf
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, GlobalAveragePooling2D


class BuildModel:
    def __init__(self, num_classes, input_shape) -> None:
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None

    def build_model(self, key):
        if key == "simple_model":
            try:
                logging.info("Building Simple Model....")
                self.model = Sequential()
                self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
                self.model.add(MaxPooling2D((2, 2)))
                self.model.add(Conv2D(64, (3, 3), activation='relu'))
                self.model.add(MaxPooling2D((2, 2)))
                self.model.add(Conv2D(128, (3, 3), activation='relu'))
                self.model.add(MaxPooling2D((2, 2)))
                self.model.add(Conv2D(256, (3,3), activation='relu'))
                self.model.add(MaxPooling2D((2, 2)))
                self.model.add(Flatten())
                self.model.add(Dense(512, activation='relu'))
                self.model.add(Dense(self.num_classes, activation='sigmoid'))

                return self.model
            
            except Exception as e:
                logging.error(f"An error occured: {str(e)}")
                raise e
            
        elif key == "res_model":
            try:
                logging.info("Building ResNet Model...")
                base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
                self.model = Sequential()
                self.model.add(base_model)
                self.model.add(GlobalAveragePooling2D())
                self.model.add(Flatten())
                self.model.add(Dense(1024, activation='relu'))
                self.model.add(Dense(self.num_classes, activation='sigmoid'))

                return self.model
            
            except Exception as e:
                logging.error(f"An error occurrd: {str(e)}")

        else:
            return None

