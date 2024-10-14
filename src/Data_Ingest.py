import numpy as np
import cv2 as cv
import tensorflow as tf
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataIngest:
    def __init__(self, data_path, image_width, image_height, batch_size):
        self.data_path = data_path
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size

    def load_data(self):
        try:
            logging.info("Building Image Generator...")
            self.train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.2
            )

            self.test_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )

        except Exception as e:
            logging.error(f"Error while building image generator: {e}")
            return None
        
    def generate_data(self):
        try:
            logging.info("Generating Images....")
            self.train_df = self.train_datagen.flow_from_directory(
                self.data_path,
                target_size=(self.image_width, self.image_height),
                batch_size=self.batch_size,
                class_mode='binary',
                subset='training',
                save_to_dir='augmented_images/',  
                save_prefix='aug',
                save_format='png'
            )

            self.test_df = self.test_datagen.flow_from_directory(
                self.data_path,
                target_size=(self.image_width, self.image_height),
                batch_size=self.batch_size,
                class_mode='binary',
                subset='validation'
            )

            return self.train_df, self.test_df

        except FileNotFoundError:
            logging.error("File not found")
            return None

        except Exception as e:
            logging.error(f"Error while generating images: {e}")
            return None
        
