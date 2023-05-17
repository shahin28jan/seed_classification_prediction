import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)

                  
class CustomData:
    def __init__(self,
                 area:float,
                 perimeter:float,
                 compactness:float,
                 length_kernel:float,
                 width_kernel:float,
                 asymmetry_coeff:float,
                 length_of_kernel_groove:float):
        
        self.area=area
        self.perimeter=perimeter
        self.compactness=compactness
        self.length_kernel=length_kernel
        self.width_kernel=width_kernel
        self.asymmetry_coeff=asymmetry_coeff
        self.length_of_kernel_groove = length_of_kernel_groove
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'area':[self.area],
                'perimeter':[self.perimeter],
                'compactness':[self.compactness],
                'length_kernel':[self.length_kernel],
                'width_kernel':[self.width_kernel],
                'asymmetry_coeff':[self.asymmetry_coeff],
                'length_of_kernel_groove':[self.length_of_kernel_groove]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
