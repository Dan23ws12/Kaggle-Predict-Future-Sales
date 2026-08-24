import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    new_data = new_data.drop(columns=["date"])
    return data_preprocessor.split_data(new_data)

def train_random_forest(n_estimators: int, min_samples_leaf: int, max_depth: Optional[int] = None):
    rand_forest = RandomForestRegressor(random_state=RAND_STATE, n_jobs=-1, 
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, 
        max_depth=max_depth, verbose=1)
    return rand_forest

