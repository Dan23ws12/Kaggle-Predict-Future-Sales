import pandas as pd
import os
from dotenv import load_dotenv
import psycopg

#importing the .env variables
load_dotenv()

def data_extraction()-> dict[str, pd.DataFrame]:
    """ 
    This function returns a dictionary containing the dataframes
    that represent the original (imported from Kaggle) tables used to train the
    prediction models
    """
    
    training_data = pd.read_csv(os.getenv('ORIGINAL_DATA_PATH') + '/sales_train.csv')
    items_data = pd.read_csv(os.getenv('ORIGINAL_DATA_PATH') + '/items.csv')
    shops_data = pd.read_csv(os.getenv('ORIGINAL_DATA_PATH') + '/shops.csv')
    test_data = pd.read_csv(os.getenv('ORIGINAL_DATA_PATH') + '/test.csv')
    items_categories_data = pd.read_csv(os.getenv('ORIGINAL_DATA_PATH') + '/item_categories.csv')
    return {
        "orig_train_df": training_data,
        "items_df": items_data,
        "item_categories_df": items_categories_data,
        "shops_df": shops_data,
        "test_df": test_data
    }
