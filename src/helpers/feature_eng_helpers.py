import pandas as pd
from .data_preprocessor import SalesDataPreprocessor
from . import TARGET_COL

data_preprocessor = SalesDataPreprocessor()

def preprocess_and_split_data(data: pd.DataFrame) -> tuple[pd.DataFrame]:
    """
    Preprocesses the sales training data by replacing infrequent values and 
    splitting the data into training and test subsets.
    Returns a tuple containing the training and test subsets data
    """
    data = data_preprocessor.replace_infrequent_values(data)
    data = data_preprocessor.add_features(data)
    keys = ["shop_id", "item_id", "month_block_num"]
    #aggregate to drop daily item_price and item_cnt_day rows and
    # to reduce data size
    data_grouped = data.groupby(keys, as_index=False).agg({
        TARGET_COL: "first",
        "month_block_length": "first",
        "item_name_length": "first",
        "item_months_sold": "first",
        "avg_item_price_per_month": "first",
        "avg_sales_per_shop": "first",
        "avg_sales_per_item": "first",
        "item_price_mean": "mean", 
        "item_price_median": "median",
    })
    return data_preprocessor.split_data(data_grouped)

def get_preprocessed_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the sales training data by replacing infrequent values and 
    adding features.
    Returns the preprocessed data.
    """
    
    new_data = data_preprocessor.replace_infrequent_values(data)
    return data_preprocessor.add_features(new_data)


