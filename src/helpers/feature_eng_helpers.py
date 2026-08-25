import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from .data_preprocessor import SalesDataPreprocessor
from . import RAND_STATE
from typing import Optional

data_preprocessor = SalesDataPreprocessor()

def preprocess_and_split_data(data: pd.DataFrame) -> tuple[pd.DataFrame]:
    """
    Preprocesses the sales training data by replacing infrequent values and 
    splitting the data into training and test subsets.
    Returns a dictionary containing the training and test subsets data
    """
    new_data = data.copy()
    new_data = data_preprocessor.replace_infrequent_values(new_data)
    new_data = data_preprocessor.add_features(new_data)
    return data_preprocessor.split_data(new_data)

def train_random_forest(n_estimators: int, min_samples_leaf: int, max_depth: Optional[int] = None):
    rand_forest = RandomForestRegressor(random_state=RAND_STATE, 
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, 
        max_depth=max_depth, verbose=1)
    rfe = RFECV(estimator=rand_forest, step=1, cv=1000, scoring=None, min_features_to_select=1,verbose=0, n_jobs=-1)
    return rfe

