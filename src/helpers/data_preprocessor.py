import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from . import CAT_FEATURES, NUMERIC_FEATURES, RAND_STATE, TARGET_COL

class SalesDataPreprocessor:
    def __init__(self):
        self.top_vals_by_col = {"shop_id": None, "item_id": None}
        self.increments_by_col = {"shop_id": 2, "item_id": 1000}
        self.isInitialized = False
        self._items_df = None
        load_dotenv()
    
    
    def initialize(self, sales_df: pd.DataFrame):
        """
        Initializes the SalesDataTransformer by getting the 
        top values that make up 90% of the column for 
        the shop_id and item_id columns.
        This is to avoid recalculating the most frequent values 
        for the shop_id and item_id columns for each training set.
        """
        self.top_vals_by_col["shop_id"] = self.get_top_vals_by_col(sales_df, 
            "shop_id", self.increments_by_col["shop_id"])
        self.top_vals_by_col["item_id"] = self.get_top_vals_by_col(sales_df, 
            "item_id", self.increments_by_col["item_id"])
        self.isInitialized = True

    def get_top_vals(self) -> pd.DataFrame:
        """
        Returns the top values that make up 90% of the column for 
        the shop_id and item_id columns.
        """
        return self.top_vals_by_col
        
    def split_data(self, sales_df: pd.DataFrame, one_hot_encode: bool = True):
        """
        Splits the sales training data into training and test subsets, and returns 
        a tuple containing the training and test subsets data
        """
        scalable_numeric_features = [feature for feature in NUMERIC_FEATURES if feature != "month_block_num"]
        col_transformer = ColumnTransformer(
            [("one hot encoding", OneHotEncoder(), CAT_FEATURES)]
            if one_hot_encode else []
        )
        if not one_hot_encode:
            for feature in CAT_FEATURES:
                sales_df[feature] = pd.Categorical(sales_df[feature])
        train_df = sales_df.drop(columns=[TARGET_COL, "date"])
        train_df = self.downgrade_numeric(train_df)
        train_df = col_transformer.fit_transform(train_df)
        #Splitting data into train and test splits
        x_train, x_test, y_train, y_test = train_test_split(train_df, 
                sales_df[TARGET_COL], test_size=0.3, random_state=RAND_STATE)
        return x_train, x_test, y_train, y_test
    
    def get_top_vals_by_col(self, df: pd.DataFrame, colname: str, increment: int) -> pd.DataFrame:
        """
        Returns a DataFrame of shop IDs and their frequencies sorted in descending order,
        limited to the point where the cumulative sum of frequencies is >= 90% of total records.
        Iterates over the dataframe in increments of the given increment.
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

    def replace_vals_by_col(self, df: pd.DataFrame, colname: str, top_vals_sr: pd.Series) -> pd.DataFrame:
        """
        Returns the df where values in colname that are not present in top_vals_df
        are replaced with "other".
        If colname is not a column in df, raises a ValueError.
        """
        if (colname not in df.columns):
            raise ValueError(f"{colname} is not a column in the DataFrame")
        df[colname] = df[colname].where(df[colname].isin(top_vals_sr), "other")
        df[colname] = df[colname].astype(str)
        return df

    def add_month_length(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the sales training dataframe with a month_block_length column
        added, representing the difference between the latest and earliest date in days in each
        month block.
        """
        month_dates = sales_df[["date", "month_block_num"]]
        month_dates["date"] = pd.to_datetime(month_dates["date"])
        def month_length_days(s):
            return (s.max() - s.min()).days
        month_dates = month_dates.groupby("month_block_num").agg(month_length_days)
        sales_df["month_block_length"] = sales_df["month_block_num"].map(month_dates["date"])
        return sales_df

    def add_item_name_length(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the sales training dataframe with an item_name_length column
        added, representing the number of alphanumeric characters in the item name
        associated with each item_id (whitespace and special characters are excluded).
        Rows with item_id "other" are assigned a length of 5.
        """
        if self._items_df is None:
            self._items_df = pd.read_csv(os.getenv("ORIGINAL_DATA_PATH") + "/items.csv")
        name_lengths = (
            self._items_df.set_index("item_id")["item_name"]
            .map(lambda name: sum(1 for c in str(name) if c.isalnum()))
        )

        def get_item_name_length(item_id) -> int:
            if str(item_id) == "other":
                return 5
            lookup_id = int(item_id) if isinstance(item_id, str) and item_id.isdigit() else item_id
            return name_lengths[lookup_id]

        sales_df["item_name_length"] = sales_df["item_id"].apply(get_item_name_length)
        return sales_df

    def add_item_months_sold(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the sales training dataframe with an item_months_sold column
        added, representing the number of distinct months during which each item was sold.
        """
        
        months_sold = sales_df.groupby("item_id")["month_block_num"].nunique()
        sales_df["item_months_sold"] = sales_df["item_id"].map(months_sold)
        return sales_df

    def add_avg_item_price_per_month(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a copy of the sales training dataframe with an avg_item_price_per_month column
        added, representing the average item price for each item in each month block.
        """
        avg_item_price_per_month = (
            sales_df.groupby(["item_id", "month_block_num"])["item_price"].transform("mean")
        )
        sales_df["avg_item_price_per_month"] = sales_df[["item_id", "month_block_num"]].map(avg_item_price_per_month)
        return sales_df

    def add_avg_sales_per_shop(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the sales training dataframe with an avg_sales_per_shop column
        added, representing the average number of sales per shop for each month block.
        """
        avg_sales_per_shop = (
            sales_df.groupby(["month_block_num", "shop_id"])[TARGET_COL]
            .mean()
        )
        sales_df["avg_sales_per_shop"] = sales_df[["month_block_num", "shop_id"]].map(avg_sales_per_shop)
        return sales_df

    def add_avg_sales_per_item(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the sales training dataframe with an avg_sales_per_item column
        added, representing the average number of sales per item for each month block.
        """
        avg_sales_per_item = (
            sales_df.groupby(["month_block_num", "item_id"])[TARGET_COL]
            .mean()
        )
        sales_df["avg_sales_per_item"] = sales_df[["month_block_num", "item_id"]].map(avg_sales_per_item)
        return sales_df

    def add_features(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the sales training dataframe with all engineered feature columns added.
        """
        new_df = self.add_month_length(sales_df)
        new_df = self.add_item_name_length(new_df)
        new_df = self.add_item_months_sold(new_df)
        new_df = self.add_avg_item_price_per_month(new_df)
        new_df = self.add_avg_sales_per_shop(new_df)
        new_df = self.add_avg_sales_per_item(new_df)
        return new_df

    def downgrade_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the dataframe with int64 columns converted to int32 and float64 columns
        converted to float32.
        """
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = df[col].astype("int32")
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype("float32")
        return df

    def replace_infrequent_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the sales training dataset df where 
        values of categorical features that are not in the top 90% of the dataset
        are replaced with "other".
        """
        if (not self.isInitialized):
            self.initialize(df)
        for col in CAT_FEATURES:
            top_vals_df = self.top_vals_by_col.get(col)
            if (top_vals_df is not None):
                df = self.replace_vals_by_col(df, col, top_vals_df[col])
        return df