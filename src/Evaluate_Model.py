import logging
from sklearn.metrics import classification_report, confusion_matrix


class Evaluation:
    def __init__(self, model, test_df) -> None:
        self.model = model
        self.test_df = test_df

    def predict_data(self):
        if self.test_df is not None:
            try:
                logging.info("Predicting Data....")
                self.y_true = self.test_df.classes
                self.y_pred = self.model.predict(self.test_df)
                self.y_pred = (self.y_pred > 0.5).astype(int)

            except Exception as e:
                logging.error(f"An error occurred:{str(e)}")
                raise e
            
    def class_report(self):
        try:
            self.report = classification_report(self.y_pred, self.y_true)
            return self.report
        
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")

    def conf_mat(self):
        try:
            self.conf = confusion_matrix(self.y_pred, self.y_true)
            return self.conf

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")