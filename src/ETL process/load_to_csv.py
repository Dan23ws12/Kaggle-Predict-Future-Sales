import pandas as pd
import os
from dotenv import load_dotenv
from data_transform import data_extraction, get_clean_data_template, get_filled_sales_data, get_full_sales_data

load_dotenv()

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
    # creates template for training data
    sales_df_template = get_clean_data_template(orig_sales_data)
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
        sales_filled_df = get_filled_sales_data(sales_df_template, fill_method=key)
        if key == "zero":
            load_full_data_to_csv(
                get_full_sales_data(sales_filled_df, items_data), 
                "sales_full_0"
            )
        # item_cnt_month is the sum of item_cnt_day; item_price_median is the
        # median item_price. Daily item_cnt_day and item_price columns are dropped.
        sales_filled_df = sales_filled_df.groupby(
            ["shop_id", "item_id", "month_block_num"], as_index=False
        ).agg(
            item_cnt_month=("item_cnt_day", "sum"),
            item_price_median=("item_price", "median"),
        )
        
        load_train_data_to_csv(sales_filled_df, value)
    