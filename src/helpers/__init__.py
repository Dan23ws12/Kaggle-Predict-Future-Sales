# categorical columns used (for model training purposes)
CAT_FEATURES:list[str] = ["shop_id", "item_id"]
# categorical columns used (for exploratory analysis purposes)
CAT_COLS:list[str] = ["month_block_num", "shop_id", "item_id", "item_category_id"]
# numeric columns used (for exploratory analysis purposes)
NUMERIC_COLS:list[str] = ["item_price", "item_cnt_month", "item_cnt_day", 
    "item_price_mean", "item_price_median"]
# numeric columns used (for model training purposes)
NUMERIC_FEATURES:list[str] = [
    "month_block_num", "month_block_length", 
    "item_name_length", "item_months_sold", 
    "avg_item_price_per_month", "item_price_median"
]
# random state used for splitting the data into training and test sets and training the models
RAND_STATE:int = 35
# target column used for model training
TARGET_COL:str = "item_cnt_month"



__all__ = [
    'epa_helpers', "feature_eng_helpers", "CAT_FEATURES", "CAT_COLS", 
    "NUMERIC_COLS", "NUMERIC_FEATURES", "RAND_STATE", "TARGET_COL", 
    "NON_TRAINING_COLS"
    ]