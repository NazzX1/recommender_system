import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from utils.exception import CustomException
from utils.logger import logging
import os

from utils.utils import save_object, LabelEncoderTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('data',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        
        try:
            numerical_columns = ['rating_counts']
            categorical_columns = [
                                "user_id",
                                "product_id",
                                 ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("cat_pipelines",cat_pipeline,categorical_columns),
                ("num", num_pipeline, numerical_columns),
                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            
            # Obtain preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Select features for transformation
            features = ["user_id", "product_id", "rating_counts"]
            train_features = train_df[features]
            test_features = test_df[features]

            # Fit and transform the training data
            logging.info("Fitting and transforming training data...")
            train_transformed = preprocessing_obj.fit_transform(train_features)

            # Transform the test data
            logging.info("Transforming testing data...")
            test_transformed = preprocessing_obj.transform(test_features)

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            return (
                train_transformed, 
                test_transformed, 
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)