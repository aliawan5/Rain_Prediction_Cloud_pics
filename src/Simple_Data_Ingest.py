import os
import cv2 as cv
import logging


class SimpleDataIngestion:
    def __init__(self, data_path) -> None:
        self.data_path = data_path

    
    def ingest_data(self):
        self.images = []
        self.labels = []
        if self.data_path:
            logging.info("Loading data through")
