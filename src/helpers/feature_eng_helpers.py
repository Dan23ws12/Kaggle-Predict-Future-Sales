import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from . import CAT_FEATURES, NUMERIC_FEATURES, RAND_STATE, TARGET_COL


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame]:
    """
    Splits the sales training data into training and test subsets, and returns 
    a dictionary containing the training and test subsets data
    """
    col_transformer = ColumnTransformer([("numeric col z scaling", StandardScaler(), NUMERIC_FEATURES),
        ("one hot encoding", OneHotEncoder(), CAT_FEATURES)]
    )
    train_df = col_transformer.fit_transform(data.drop(columns=[TARGET_COL]))
    #Splitting data into train and test splits
    x_train, x_test, y_train, y_test = train_test_split(train_df, 
            data[TARGET_COL], test_size=0.3, random_state=RAND_STATE)
    return x_train, x_test, y_train, y_test

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

def get_top_by_col(df: pd.DataFrame, colname: str, increment: int) -> pd.DataFrame:
    """
    Returns a DataFrame of shop IDs and their frequencies sorted in descending order,
    limited to the point where the cumulative sum of frequencies is >= 90% of total records.
    Iterates in increments of 2.
    """
    frequency_df = df[colname].value_counts().reset_index()
    frequency_df.columns = [colname, 'frequency']
    frequency_df = frequency_df.sort_values(by='frequency', ascending=False).reset_index(drop=True)
    
    total_records = len(df)
    target = 0.9 * total_records
    cumsum, last_index = 0, 0
    for i in range(0, len(frequency_df), increment):
        # first iteration, added so that the cumulative sum 
        # is calculated correctly and the function doesn't have to 
        # sum from the beginning of the dataframe each time
        if (last_index == 0):
            cumsum = frequency_df.loc[0:i, 'frequency'].sum()
            last_index = i
        else:
            cumsum += frequency_df.loc[last_index:i, 'frequency'].sum() # cumulative sum of frequencies
            last_index = i # last index of the cumulative sum
        if cumsum >= target:
            return frequency_df.iloc[: i + 1]
            
    return frequency_df

def replace_values(df: pd.DataFrame, colname: str, ref_series: pd.Series) -> pd.DataFrame:
    """
    Returns a copy of df where values in colname that are not present in ref_series
    are replaced with "other".
    If colname is not a column in df, raises a ValueError.
    """
    if (colname not in df.columns):
        raise ValueError(f"{colname} is not a column in the DataFrame")
    new_df = df.copy()
    new_df[colname] = new_df[colname].where(new_df[colname].isin(ref_series), "other")
    return new_df

def replace_infrequent_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of a sales training dataset df where 
    values of categorical features that are not in the top 90% of the dataset
    are replaced with "other".
    """
    # Increments by column for getting the top 90% of values
    # this is to avoid the case where a column has a lot of unique values
    # only shop_id and item_id can have values replaced with "other"
    increments_by_col = {
        "shop_id": 2,
        "item_id": 1000
    }
    new_df = df.copy()
    for col in CAT_FEATURES:
        increment = increments_by_col.get(col, 2)
        if (increment):
            top_90 = get_top_by_col(new_df, col, increment)
            new_df = replace_values(new_df, col, top_90[col])
        
        
    return new_df