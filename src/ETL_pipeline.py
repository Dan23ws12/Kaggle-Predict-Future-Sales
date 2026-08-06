import pandas as pd
import os
from dotenv import load_dotenv



def data_extraction(table_name: str)-> pd.DataFrame:
    """ 
    This function returns a dataframe of the table_name that represent the original (imported from Kaggle) tables used to train the
    prediction models
    """
    
    if table_name == "sales_train":
        return pd.read_csv(os.getenv('ORIGINAL_DATA_PATH') + '/sales_train.csv')
    elif table_name == "items":
        return pd.read_csv(os.getenv('ORIGINAL_DATA_PATH') + '/items.csv')
    elif table_name == "shops":
        return pd.read_csv(os.getenv('ORIGINAL_DATA_PATH') + '/shops.csv')
    elif table_name == "test":
        return pd.read_csv(os.getenv('ORIGINAL_DATA_PATH') + '/test.csv')
    elif table_name == "item_categories":
        return pd.read_csv(os.getenv('ORIGINAL_DATA_PATH') + '/item_categories.csv')
    else:
        raise ValueError(f"Table {table_name} not found")

def get_full_sales_data(sales: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of the sales_train dataframe with the date column, item_category_id and item_id columns
    turned into a datetime object and the monthly aggregated training table added as
    the item_cnt_month column
    """
    # converting the sales dataframe
    sales_df = sales.copy(deep=False)
    sales_df["date"] = pd.to_datetime(sales_df["date"], format="%d.%m.%Y")

    # merging the training data with items dataframe
    sales_df = pd.merge(sales_df, items, on="item_id", how="inner")
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
        mean_val = sales_train_df["item_cnt_day"].mean()
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
    sales_train_df = sales.copy(deep=False)
    # removing duplicate rows (no duplicate rows in other tables)
    sales_train_df.drop_duplicates(inplace=True)
    sales_train_df.rename(columns={"date_block_num": "month_block_num"}, inplace=True)
    # dropping date column as it is no longer needed
    sales_train_df.drop(columns=["date"], inplace=True)
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


def load_full_data_to_csv(sales: pd.DataFrame, table_name:str):
    """This function loads the full dataset to the csv files"""
    sales.to_csv(os.getenv('FULL_DATA_PATH') + '/' + table_name + '.csv', index=False)

def load_train_data_to_csv(train_data: pd.DataFrame, table_name:str):
    """This function loads the training dataset to the csv files"""
    train_data.to_csv(os.getenv('CLEAN_DATA_PATH') + '/' + table_name + '.csv', index=False)


if __name__ == "__main__":
    #importing the .env variables
    load_dotenv()
    # imports original sales data
    orig_sales_data = data_extraction("sales_train")
    items_data = data_extraction("items")[["item_id", "item_category_id"]]   
    #loads data for exploration and validation into csv
    load_full_data_to_csv(get_full_sales_data(orig_sales_data, items_data), "sales_full_0")
    # creates template for training data
    sales_df_template = get_train_data_template(orig_sales_data)
    # names of dataframes to be created
    df_table_names = {
        "zero": "sales_train_cnt_0",
        "abs": "sales_train_cnt_abs",
        "mean": "sales_train_cnt_mean",
        "bfill": "sales_train_cnt_bfill",
        "ffill": "sales_train_cnt_ffill"
    }
    #loops through the names of the dataframes and loads the training data to csv
    for key, value in df_table_names.items():
        load_train_data_to_csv(get_sales_train_data(sales_df_template, fill_method=key), value)
    
    
    
