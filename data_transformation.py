import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler, LabelEncoder

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
                ("encoder",LabelEncoderTransformer()),
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("cat_pipelines",cat_pipeline,categorical_columns)
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

            preprocessing_obj=self.get_data_transformer_object()
            train_columns = ["user_id",
                            "product_id",
                            "rating",
                            "rating_count",
                            "category",
                            "product_name",
                            "img_link",
                            "about_product"
                          ]
            numerical_columns = ["rating", "rating_count"]

  
        
                
            # input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            # target_feature_train_df=train_df[target_column_name]

            # input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            # target_feature_test_df=test_df[target_column_name]
           



            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            print(train_df)
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            print(input_feature_train_arr)
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            return (
                train_arr,
                # train_df,
                test_arr,
                # test_df,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)