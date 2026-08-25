import pandas as pd
import os
from dotenv import load_dotenv

ACCEPTABLE_TABLE_NAMES = ["sales_train", "items", "shops", "test", "item_categories"]

def data_extraction(table_name: str)-> pd.DataFrame:
    """ 
    This function returns a dataframe of the table_name that represent the original (imported from Kaggle) tables used to train the
    prediction models
    """
    
    if table_name in ACCEPTABLE_TABLE_NAMES:
        return pd.read_csv(os.getenv('ORIGINAL_DATA_PATH') + '/' + table_name + '.csv')
    else:
        raise ValueError(f"Table {table_name} not found")

def get_full_sales_data(sales: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of the sales_train dataframe (with negative values in 
    item_cnt_day replaced by 0) with the date column removed, 
    item_category_id remains and added aggregated item_cnt_day by
    month, shop and item as item_cnt_month column
    """
    # sales argument already has negative values and date removed
    # and item_cnt_month calculated
    # merging the training data with items dataframe to add item_category_id
    sales_df = pd.merge(sales, items, on="item_id", how="inner")
    # returning the aggregated item_cnt_day by month, 
    # shop and item as item_cnt_month column
    return sales_df
    

def get_sales_train_data(sales:pd.DataFrame, fill_method="zero"):
    """
    Returns a copy of the sales dataframe for training with negative values in 
    item_cnt_day replaced by the desired fill_method
    and item_cnt_day is then replaced with item_cnt_month
    """
    sales_train_df = sales.copy(deep=False)
    if fill_method == "zero":
        sales_train_df["item_cnt_day"] = sales_train_df["item_cnt_day"].map(lambda x: x if x>=0 else 0)
    elif fill_method == "abs":
        sales_train_df["item_cnt_day"] = sales_train_df["item_cnt_day"].abs()
    elif fill_method == "mean":
        mean_val = sales_train_df.loc[sales_train_df["item_cnt_day"] >= 0, "item_cnt_day"].mean()
        sales_train_df["item_cnt_day"] = sales_train_df["item_cnt_day"].map(lambda x: x if x >= 0 else mean_val)
    elif fill_method == "bfill":
        sales_train_df["item_cnt_day"] = sales_train_df["item_cnt_day"].mask(sales_train_df["item_cnt_day"] < 0)
        sales_train_df["item_cnt_day"]= sales_train_df["item_cnt_day"].bfill()
    elif fill_method == "ffill":
        sales_train_df["item_cnt_day"] = sales_train_df["item_cnt_day"].mask(sales_train_df["item_cnt_day"] < 0)
        sales_train_df["item_cnt_day"]= sales_train_df["item_cnt_day"].ffill()
    else:
        raise ValueError(f"fill_method {fill_method} not found")
    return get_item_cnt_month(sales_train_df)

    

def get_train_data_template(sales: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of the sales dataframe with the date column removed, 
    date_block_num renamed to month_block_num and item_price 
    turned into its absolute value (only one value is negative)
    This function returns a remplate so that other functions can 
    fill in item_cnt_day with the desired values
    """
    # removing duplicate rows (no duplicate rows in other tables)
    sales_train_df = sales.drop_duplicates()
    sales_train_df.rename(columns={"date_block_num": "month_block_num"}, inplace=True)
    sales_train_df["date"] = pd.to_datetime(sales_train_df["date"], format="%d.%m.%Y")
    # replacing the item_price with the absolute value
    sales_train_df["item_price"] = sales_train_df["item_price"].map(lambda x: abs(x))
    return sales_train_df
    

def get_item_cnt_month(df: pd.DataFrame)-> dict[str, pd.DataFrame]:
    """
    This function calculates total sales per month for each item and store
    and returns a copy of the original dataframe with the total named item_cnt_month
    """
    # gets the total number of items sold per month and per store
    item_cnt_month = df.groupby(["item_id", "shop_id", "month_block_num"])["item_cnt_day"].sum().reset_index()
    #renames the sum to item_cnt_month
    item_cnt_month.rename(columns={"item_cnt_day": "item_cnt_month"}, inplace=True)
    # removes original column
    df.drop(columns = ["item_cnt_day"], inplace=True)
    # merges with original database 
    df = df.merge(item_cnt_month, on=["item_id", "shop_id", "month_block_num"], how="left")
    return df

    
