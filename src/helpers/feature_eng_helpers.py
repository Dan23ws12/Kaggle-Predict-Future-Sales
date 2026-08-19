import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from .data_preprocessor import SalesDataPreprocessor
from . import CAT_FEATURES, NUMERIC_FEATURES, RAND_STATE, TARGET_COL

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

def train_random_forest(x_train, y_train):
    param_grid = {
        "n_estimators": [70, 100, 150],
        "min_samples_leaf": [1, 2, 5],
        "max_depth": [None, 9, 11]
    }

    rand_forest = RandomForestRegressor(random_state=RAND_STATE)

    grid_search = GridSearchCV(estimator=rand_forest, param_grid=param_grid, 
        cv=10, n_jobs=-1
    )
    grid_search.fit(x_train, y_train)
    print(f"best score of cv is {grid_search.best_score_}")
    print("best parameters")
    print(grid_search.best_params_)
    return grid_search.best_estimator_

