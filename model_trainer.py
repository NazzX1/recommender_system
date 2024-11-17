import os
import sys
from dataclasses import dataclass
from models import *

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras 
import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils.exception import CustomException
from utils.logger import logging

from utils.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("data", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            print(X_train.shape)
            models = {
                "Content_Based_model":Content_Based_model(),
                "Colaborative_Filtering_model": Colaborative_Filtering_model(X_train, y_train)
            }

            model_report = []

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")
                
                initial_learning_rate = 0.01
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate,
                    decay_steps=100,
                    decay_rate=0.96,
                    staircase=True
                )
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
                loss = tf.losses.MeanAbsoluteError()

                model_report.append(evaluate_models(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    models={model_name: model},
                    optimizer=optimizer,
                    loss=loss
                ))
            best_model_info = min(model_report, key=lambda x: list(x.values())[0])
            best_model_name = list(best_model_info.keys())[0]
            best_model_score = list(best_model_info.values())[0]
            best_model = models[best_model_name]
            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")


            best_model.save("data/model_data.h5")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict([X_test[:, 0], X_test[:, 1]])

            r2_square = mean_squared_error(y_test, predicted)
            return r2_square
            

        except Exception as e:
            raise CustomException(e,sys)