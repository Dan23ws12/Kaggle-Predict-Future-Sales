import pandas as pd
from .data_preprocessor import SalesDataPreprocessor

data_preprocessor = SalesDataPreprocessor()

def preprocess_and_split_data(data: pd.DataFrame) -> tuple[pd.DataFrame]:
    """
    Preprocesses the sales training data by replacing infrequent values and 
    splitting the data into training and test subsets.
    Returns a tuple containing the training and test subsets data
    """
    data = data_preprocessor.replace_infrequent_values(data)
    data = data_preprocessor.add_features(data)
    return data_preprocessor.split_data(data)

def get_preprocessed_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the sales training data by replacing infrequent values and 
    adding features.
    Returns the preprocessed data.
    """
    
    new_data = data_preprocessor.replace_infrequent_values(data)
    return data_preprocessor.add_features(new_data)


