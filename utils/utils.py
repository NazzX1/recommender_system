import os
import sys

import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from utils.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, optimizer, loss):
    try:
        report = {}

        for model_name, model in models.items():
            
            model.compile(optimizer=optimizer, loss=loss)

            model.fit(
                [X_train[:, 0], X_train[:, 1]], y_train,
                validation_split=0.2,  
                epochs=1, 
                batch_size=48,  
                verbose=1  
            )

            y_train_pred = model.predict([X_train[:, 0], X_train[:, 1]])
            y_test_pred = model.predict([X_test[:, 0], X_test[:, 1]])

            train_model_score = mean_squared_error(y_train, y_train_pred)
            test_model_score = mean_squared_error(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(f"Error in model evaluation: {str(e)}", sys)

    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    



class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.encoders = {}
        for column in X.columns:
            if X[column].dtype == 'object':
                le = LabelEncoder()
                le.fit(X[column])
                self.encoders[column] = le
        return self

    def transform(self, X):
        X_copy = X.copy()
        for column, encoder in self.encoders.items():
            if column in X_copy.columns:
                X_copy[column] = encoder.transform(X_copy[column])
        return X_copy