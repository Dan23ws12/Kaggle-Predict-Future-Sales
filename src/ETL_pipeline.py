import pandas as pd
import os
from dotenv import load_dotenv



def data_extraction()-> dict[str, pd.DataFrame]:
    """ 
    This function returns a dictionary containing the dataframes
    that represent the original (imported from Kaggle) tables used to train the
    prediction models
    """
    
    sales_train_data = pd.read_csv(os.getenv('ORIGINAL_DATA_PATH') + '/sales_train.csv')
    items_data = pd.read_csv(os.getenv('ORIGINAL_DATA_PATH') + '/items.csv')
    shops_data = pd.read_csv(os.getenv('ORIGINAL_DATA_PATH') + '/shops.csv')
    test_data = pd.read_csv(os.getenv('ORIGINAL_DATA_PATH') + '/test.csv')
    items_categories_data = pd.read_csv(os.getenv('ORIGINAL_DATA_PATH') + '/item_categories.csv')
    return {
        "orig_sales_train": sales_train_data,
        "items": items_data,
        "shops": shops_data,
        "test": test_data,
        "item_categories": items_categories_data
    }

def data_transformation(data_dict: dict[str, pd.DataFrame])-> dict[str, pd.DataFrame]:
    """
    This function returns a dictionary of copied dataframes of the sales_train dataframe
    with different transformations done to them.
    """
    sales_train_df = data_dict["orig_sales_train"].copy(deep=False)
    sales_train_df.drop_duplicates(inplace=True)
    # removing duplicate rows (no duplicate rows in other tables)
    sales_train_df.rename(columns={"date_block_num": "month_block_num"}, inplace=True)
    # dropping date column as it is no longer needed
    sales_train_df.drop(columns=["date"], inplace=True)
    # subsetting items dataframe to only include item_id and item_category_id columns 
    # to reduce memory usage
    item_categories = data_dict["items"].loc[:, ["item_id", "item_category_id"]]
    sales_train_df = sales_train_df.merge(item_categories, on="item_id", how="inner")

    #turns id columns and month_block_num to string for easier one-hot encoding
    for col in ["item_id", "shop_id", "item_category_id", "month_block_num"]:
        sales_train_df[col] = sales_train_df[col].map(str)    

    # replacing the item_price with the absolute value
    sales_train_df["item_price"] = sales_train_df["item_price"].map(lambda x: abs(x))
    
    # making data frames for different choices on how to change item_cnt_day
    #replace negative values with 0
    sales_train_cnt_0 = sales_train_df.copy(deep=False)
    sales_train_cnt_0["item_cnt_day"] = sales_train_cnt_0["item_cnt_day"].map(lambda x: 0 if x < 0 else x)

    # replace negative values with the absolute value
    sales_train_cnt_abs = sales_train_df.copy(deep=False)
    sales_train_cnt_abs["item_cnt_day"] = sales_train_cnt_abs["item_cnt_day"].map(lambda x: abs(x))

    # replace negative values with mean
    sales_train_cnt_mean = sales_train_df.copy(deep=False)
    sales_train_cnt_mean["item_cnt_day"] = sales_train_cnt_mean["item_cnt_day"].map(lambda x: x if x >= 0 else sales_train_cnt_mean["item_cnt_day"].mean())

    # use backfill to replace negative values
    sales_train_cnt_bfill = sales_train_df.copy(deep=False)
    sales_train_cnt_bfill["item_cnt_day"] = sales_train_cnt_bfill["item_cnt_day"].mask(sales_train_cnt_bfill["item_cnt_day"] < 0)
    sales_train_cnt_bfill["item_cnt_day"] = sales_train_cnt_bfill["item_cnt_day"].bfill()
    
    # use forward fill to replace negative values
    sales_train_cnt_ffill = sales_train_df.copy(deep=False)
    sales_train_cnt_ffill["item_cnt_day"] = sales_train_cnt_ffill["item_cnt_day"].mask(sales_train_cnt_ffill["item_cnt_day"] < 0)
    sales_train_cnt_ffill["item_cnt_day"] = sales_train_cnt_ffill["item_cnt_day"].ffill()
    
    
    return {
        "sales_train_cnt_0": sales_train_cnt_0,
        "sales_train_cnt_abs": sales_train_cnt_abs,
        "sales_train_cnt_mean": sales_train_cnt_mean,
        "sales_train_cnt_bfill": sales_train_cnt_bfill,
        "sales_train_cnt_ffill": sales_train_cnt_ffill
    }


        

def data_loading_transformed_dfs(dataframes: dict[str, pd.DataFrame]):
    """This function loads the transformed training tables to the csv files"""
    for key, value in dataframes.items():
        value.to_csv(os.getenv('CLEAN_DATA_PATH') + '/' + key + '.csv', index=False)

if __name__ == "__main__":
    #importing the .env variables
    load_dotenv()
    original_data = data_extraction()
    transformed_data = data_transformation(original_data)
    data_loading_transformed_dfs(transformed_data)
    
    
    
    
