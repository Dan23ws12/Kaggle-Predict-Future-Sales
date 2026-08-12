import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

cat_features = ["month_block_num", "shop_id", "item_id", "item_category_id"]
numeric_vars = ["item_price", "item_month"]
numeric_features = ["item_price"]
rand_state = 35
target_col = "item_cnt_month"

def split_data(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Splits the sales training data into training and test subsets, and returns 
    a dictionary containing the training and test subsets data
    """
    col_transformer = ColumnTransformer([("numeric col z scaling", StandardScaler(), numeric_features),
        ("one hot encoding", OneHotEncoder(), cat_features)]
    )
    train_df = col_transformer.fit_transform(data.drop(columns=[target_col]))
    #Splitting data into train and test splits
    x_train, x_test, y_train, y_test = train_test_split(train_df, 
            data[target_col], test_size=0.3, random_state=rand_state)
    return {
        "train_features" : x_train,
        "test_features": x_test,
        "train_target": y_train,
        "test_target": y_test
    }

def train_random_forest(x_train, y_train):
    param_grid = {
        "n_estimators": [70, 100, 150],
        "min_samples_leaf": [1, 2, 5],
        "max_depth": [None, 9, 11]
    }

    rand_forest = RandomForestRegressor(random_state=rand_state)

    grid_search = GridSearchCV(estimator=rand_forest, param_grid=param_grid, 
        cv=10, n_jobs=-1
    )
    grid_search.fit(x_train, y_train)
    print(f"best score of cv is {grid_search.best_score_}")
    print("best parameters")
    print(grid_search.best_params_)
    return grid_search.best_estimator_
