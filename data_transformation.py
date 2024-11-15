import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from utils.exception import CustomException
from utils.logger import logging
import os

from utils.utils import save_object, LabelEncoderTransformer



def clean_and_extract_tags(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text.lower())
    tags = [token.text for token in doc if token.text.isalnum() and token.text not in STOP_WORDS]
    return ','.join(tags)

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('data', "proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        
        try:
            numerical_columns = ['rating_count']
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
                ("cat_pipelines", cat_pipeline, categorical_columns),
                ("num", num_pipeline, numerical_columns),
                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()

            features = ["user_id", "product_id", "rating_count", 'rating', 'tags']
            
            train_df["rating"] = pd.to_numeric(train_df["rating"], errors="coerce")
            train_df["rating_count"] = train_df["rating_count"].str.replace(',', '')
            train_df["rating_count"] = pd.to_numeric(train_df["rating_count"], errors="coerce")

            test_df["rating"] = pd.to_numeric(test_df["rating"], errors="coerce")
            test_df["rating_count"] = test_df["rating_count"].str.replace(',', '')
            test_df["rating_count"] = pd.to_numeric(test_df["rating_count"], errors="coerce")

            train_df.fillna({"rating": train_df["rating"].median(), "rating_count": train_df["rating_count"].median()}, inplace=True)
            test_df.fillna({"rating": test_df["rating"].median(), "rating_count": test_df["rating_count"].median()}, inplace=True)

            column_contain_tags = ['about_product', "category"]

            train_df["about_product"] = train_df["about_product"].apply(clean_and_extract_tags)
            test_df["about_product"] = test_df["about_product"].apply(clean_and_extract_tags)


            train_df['category'] = train_df['category'].str.replace('|', ',')
            test_df['category'] = test_df['category'].str.replace('|', ',') 

            train_df['tags'] = train_df[column_contain_tags].apply(lambda x: ', '.join(x), axis=1)
            test_df['tags'] = test_df[column_contain_tags].apply(lambda x: ', '.join(x), axis=1)


            train_features = train_df[features]
            test_features = test_df[features]
            
            logging.info("Fitting and transforming training data...")
            train_transformed = preprocessing_obj.fit_transform(train_features)

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
            raise CustomException(e, sys)