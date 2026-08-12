CAT_FEATURES:list[str] = ["month_block_num", "shop_id", "item_id"]
CAT_COLS:list[str] = ["month_block_num", "shop_id", "item_id", "item_category_id"]
NUMERIC_COLS:list[str] = ["item_price", "item_cnt_month"]
NUMERIC_FEATURES:list[str] = ["item_price"]
RAND_STATE:int = 35
TARGET_COL:str = "item_cnt_month"


__all__ = ['epa_helpers', "feature_eng_helpers", "CAT_FEATURES", "NUMERIC_COLS", "NUMERIC_FEATURES", "RAND_STATE", "TARGET_COL"]