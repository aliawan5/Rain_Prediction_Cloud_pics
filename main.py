from src.Data_Ingest import DataIngest
from src.Build_Model import BuildModel
from src.Train_Model import Trainer
from src.Evaluate_Model import Evaluation


def main():
    data_path = r"C:\Users\FINE\Desktop\Rain_prediction_using_cloud_images\Data"
    image_width, image_height, batch_size = 150, 150, 32
    num_classes = 1
    key = 'simple_model'
    input_shape = (150,150,3)
    learning_rate = 0.0001
    epochs = 5

    obj1 = DataIngest(data_path, image_width, image_height, batch_size)
    obj1.load_data()
    train_df, test_df = obj1.generate_data()

    obj2 = BuildModel(num_classes, input_shape)
    model = obj2.build_model(key)

    obj3 = Trainer(learning_rate, model, epochs, train_df)
    obj3.compile_model()
    obj3.train_model()

    obj4 = Evaluation(model, test_df)
    obj4.predict_data()
    classification_report = obj4.class_report()
    confusion_matrix = obj4.conf_mat()

    print(classification_report)
    print((confusion_matrix))

if __name__ == "__main__":
    main()