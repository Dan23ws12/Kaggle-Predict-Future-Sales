import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from . import CAT_FEATURES, NUMERIC_FEATURES, RAND_STATE, TARGET_COL

class SalesDataPreprocessor:
    def __init__(self):
        self.top_vals_by_col = {"shop_id": None, "item_id": None}
        self.increments_by_col = {"shop_id": 2, "item_id": 1000}
        self.isInitialized = False
    
    
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
        
    def split_data(self, sales_df: pd.DataFrame):
        """
        Splits the sales training data into training and test subsets, and returns 
        a tuple containing the training and test subsets data
        """
        col_transformer = ColumnTransformer([
            ("numeric col z scaling", StandardScaler(), NUMERIC_FEATURES),
            ("one hot encoding", OneHotEncoder(), CAT_FEATURES)])
        train_df = col_transformer.fit_transform(sales_df.drop(columns=[TARGET_COL]))
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
        Returns a copy of df where values in colname that are not present in top_vals_df
        are replaced with "other".
        If colname is not a column in df, raises a ValueError.
        """
        if (colname not in df.columns):
            raise ValueError(f"{colname} is not a column in the DataFrame")
        new_df = df.copy()
        new_df[colname] = new_df[colname].where(new_df[colname].isin(top_vals_sr), "other")
        new_df[colname] = new_df[colname].astype(str)
        return new_df

    def replace_infrequent_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a copy of a sales training dataset df where 
        values of categorical features that are not in the top 90% of the dataset
        are replaced with "other".
        """
        
        new_df = df.copy()
        if (not self.isInitialized):
            self.initialize(new_df)
        for col in CAT_FEATURES:
            top_vals_df = self.top_vals_by_col.get(col)
            if (top_vals_df is not None):
                new_df = self.replace_vals_by_col(new_df, col, top_vals_df[col])
        return new_df