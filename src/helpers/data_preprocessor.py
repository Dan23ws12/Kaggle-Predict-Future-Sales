import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  OrdinalEncoder
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
    
    def get_top_vals_by_col(self, df: pd.DataFrame, colname: str, increment: int) -> pd.DataFrame:
        """
        Returns a DataFrame of shop IDs and their frequencies sorted in descending order,
        limited to the point where the cumulative sum of frequencies is >= 90% of total records.
        Iterates over the dataframe in increments of the given increment.
        """
        frequency_df = df[colname].value_counts().reset_index()
        frequency_df.columns = [colname, 'frequency']
        frequency_df = frequency_df.sort_values(by='frequency', ascending=False).reset_index(drop=True)
        
        target = 0.9 * len(df)
        n = len(frequency_df)
        cumsum = 0
        start = 0
        while start < n:
            end = min(start + increment - 1, n - 1)
            cumsum += frequency_df.loc[start:end, "frequency"].sum()
            if cumsum >= target:
                while end > 0 and (cumsum - frequency_df.loc[end, "frequency"]) >= target:
                    cumsum -= frequency_df.loc[end, "frequency"]
                    end -= 1
                return frequency_df.iloc[: end + 1]
            start = end + 1
                
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
        
    def split_data(self, sales_df: pd.DataFrame, one_hot_encode: bool = True):
        """
        Splits the sales training data into training and test subsets, and returns 
        a tuple containing the training and test subsets data
        """
        # only ordinal-encode the categorical features if using a model that needs it
        # aka random forest or decision tree
        sales_df = self.downgrade_numeric(sales_df)
        train_df = sales_df[NUMERIC_FEATURES + CAT_FEATURES]
        x_train, x_test, y_train, y_test = train_test_split(train_df, 
                sales_df[TARGET_COL], test_size=0.3, random_state=RAND_STATE)
        if one_hot_encode:
            ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", dtype=np.float32, unknown_value=-1)
            x_train[CAT_FEATURES] = ordinal_encoder.fit_transform(x_train[CAT_FEATURES])
            x_test[CAT_FEATURES] = ordinal_encoder.transform(x_test[CAT_FEATURES])
        else:
            for feature in CAT_FEATURES:
                x_train[feature] = pd.Categorical(x_train[feature])
                x_test[feature] = pd.Categorical(x_test[feature])
        return x_train, x_test, y_train, y_test

    def add_item_name_length(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the sales training dataframe with an item_name_length column
        added, representing the number of alphanumeric characters in the item name
        associated with each item_id (whitespace and special characters are excluded).
        Rows with item_id "other" are assigned a length of 5.
        """
        if self._items_df is None:
            self._items_df = pd.read_csv(os.getenv("ORIGINAL_DATA_PATH") + "/items.csv")
        # Count letters/digits only (same as str.isalnum); compute once per item, not per sales row.
        item_ids = self._items_df["item_id"]
        name_lengths = (
            self._items_df["item_name"].astype(str).str.count(r"[^\W_]").to_numpy()
        )
        lookup = dict(zip(item_ids.to_numpy(), name_lengths))
        lookup.update(zip(item_ids.astype(str).to_numpy(), name_lengths))
        lookup["other"] = 5
        sales_df["item_name_length"] = sales_df["item_id"].map(lookup)
        return sales_df

    def add_num_months_sold_prior(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the sales training dataframe with a num_months_sold_prior column
        added, representing the number of distinct months during which each item was sold.
        """
        
        months_sold = sales_df.groupby("item_id")["month_block_num"].nunique()
        sales_df["num_months_sold_prior"] = sales_df["item_id"].map(months_sold)
        return sales_df

    def add_avg_item_price_per_month(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a copy of the sales training dataframe with an avg_item_price_per_month column
        added, representing the average item_price_median for each item in each month block.
        """
        sales_df["avg_item_price_per_month"] = (
            sales_df.groupby(["item_id", "month_block_num"])["item_price_median"].transform("mean")
        )
        return sales_df

    def add_avg_shop_price_per_month(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a copy of the sales training dataframe with an avg_shop_price_per_month column
        added, representing the average item_price_median for each shop in each month block.
        """
        sales_df["avg_shop_price_per_month"] = (
            sales_df.groupby(["shop_id", "month_block_num"])["item_price_median"].transform("mean")
        )
        return sales_df

    def _rolling_prior_month_sales(
        self, sales_df: pd.DataFrame, group_col: str, result_col: str
    ) -> pd.DataFrame:
        """
        Adds result_col as the expanding mean of prior months' item_cnt_month
        totals for each value of group_col. The current month is excluded.
        Months with no prior history are NaN.
        """
        monthly = (
            sales_df.groupby([group_col, "month_block_num"], as_index=False)["item_cnt_month"]
            .sum()
            .sort_values([group_col, "month_block_num"], kind="mergesort")
        )
        monthly[result_col] = (
            monthly.groupby(group_col, sort=False)["item_cnt_month"]
            .transform(lambda s: s.shift(1).expanding().mean())
        )
        return sales_df.merge(
            monthly[[group_col, "month_block_num", result_col]],
            on=[group_col, "month_block_num"],
            how="left",
        )

    def add_avg_sales_per_shop(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the sales training dataframe with an avg_sales_per_shop column
        added, representing the rolling average of that shop's prior months' sales totals.
        """
        return self._rolling_prior_month_sales(sales_df, "shop_id", "avg_sales_per_shop")

    def add_avg_sales_per_item(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the sales training dataframe with an avg_sales_per_item column
        added, representing the rolling average of that item's prior months' sales totals.
        """
        return self._rolling_prior_month_sales(sales_df, "item_id", "avg_sales_per_item")

    def add_features(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the sales training dataframe with all engineered feature columns added.
        """
        new_df = self.add_item_name_length(sales_df)
        new_df = self.add_num_months_sold_prior(new_df)
        new_df = self.add_avg_shop_price_per_month(new_df)
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

    