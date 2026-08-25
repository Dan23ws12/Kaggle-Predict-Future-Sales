
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from dotenv import load_dotenv
import seaborn as sns
from . import CAT_COLS, NUMERIC_COLS, TARGET_COL
load_dotenv()

def get_df_col_and_datatypes(df: pd.DataFrame):
    """
    this function prints the shape, columns, types, description, missing values, and unique values of the DataFrame.
    """
    print(f"columns: \n {df.columns.tolist()}")
    print(f"types: \n {df.dtypes}")

def get_basic_stats(df: pd.DataFrame):
    """
    This function returns a dictionary of basic statistics for a given DataFrame.
    """
    print(f"shape: {df.shape}")
    df_numeric_cols = [col for col in df.columns if col not in CAT_COLS]
    df_categorical_cols = [col for col in df.columns if col in CAT_COLS]
    print(f"dataframe basic numeric statistics \n{df[df_numeric_cols].describe()}")
    print(f"dataframe categorical statistics \n {df[df_categorical_cols].map(lambda x: str(x)).describe()}")

def check_data_validity(df: pd.DataFrame):
    """
    This function checks the validity of the data in a given DataFrame by checking
    for missing values, invalid values and unique values
    """
    print(f"missing values: \n {df.isnull().sum()}")
    print(f"unique values: \n {df.nunique()}")
    numeric_cols = []
    non_numeric_cols = []
    for col in df.columns:
        # checks if the column contains numeric values and is not a categorical column
        if (col not in CAT_COLS) and (df[col].dtype in ['int64', 'float64']):
            numeric_cols.append(col)
        else:
            non_numeric_cols.append(col)
    # numerical values in this case are expected to be positive
    invalid_numeric_values = df[numeric_cols].map(lambda x: 1 if x<0 else 0).sum()
    # categorical values in this case are expected to be non-empty strings or numbers
    invalid_cat_values = df[non_numeric_cols].map(lambda x: 1 if ((pd.isna(x)) or (str(x)=='')) else 0).sum()
    print(f"invalid numeric values: \n {invalid_numeric_values}")
    print(f"invalid categorical values: \n {invalid_cat_values}")



def modify_categorical_columns(df:pd.DataFrame) -> pd.DataFrame:
    """
    Modifies the categorical columns in a given dataframe by concatenating
    the item category with the item id
    """
    new_df = df[NUMERIC_COLS]
    for col in CAT_COLS:
        new_df[col] = df[col].astype(str)
    return new_df

def cramer_v(col1: pd.Series, col2:pd.Series) -> float:
    """
    Returns the Cramer V correlation between two nominal variables
    """
    # Create a contingency table
    contingency_table = pd.crosstab(col1, col2)
    chi2_statistic, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Calculate Cramer's V
    n = contingency_table.sum().sum()
    phi2 = chi2_statistic / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    k_corr = k - (k - 1) * (k - 2) / (n - 1)
    r_corr = r - (r - 1) * (r - 2) / (n - 1)
    v = np.sqrt(phi2corr / min(k_corr - 1, r_corr - 1))
    
    return v

def get_nominal_var_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the Cramer V correlation matrix of all nominal variables
    in a given dataframe
    """
    # empty correlation matrix dataframe
    corr_matrix = pd.DataFrame(index=CAT_COLS, columns=CAT_COLS)
    num_cat_vars = len(CAT_COLS)
    corr_values = {}
    for k in range(num_cat_vars):
        corr_values[k] = []

    for i in range(num_cat_vars):
        for j in range(i, num_cat_vars):
            vari, varj = [CAT_COLS[i], CAT_COLS[j]]
            # matrix diagonal case (in correlation matrix)
            # m at row i and column i is ~1.00
            if (i == j):
                corr_values.get(i).append(cramer_v(df[vari], df[varj])) 
            # we haven't calculated corr(vari, varj)
            # when i > j, value at m row i and column j has been
            #appended in prior iteration
            elif (i < j):
                corr_coef = cramer_v(df[varj], df[vari])
                #store correlation coefficient  in dict
                corr_values.get(i).append(corr_coef)
                corr_values.get(j).append(corr_coef)
    
    for k in range(num_cat_vars):
        corr_matrix[CAT_COLS[k]] = corr_values.get(k)
    return corr_matrix

def get_charts_for_cat_vars(df:pd.DataFrame, colname:str):
    """
    Plots a pie chart and bar chart for a given categorical variable
    """
    if colname not in CAT_COLS:
        raise ValueError("Column is not a categorical variable")
    label_name_dict = {
        "shop_id": "Shop",
        "item_id": "Item",
        "item_category_id": "Item Category",
        "month_block_num": "Month"
    }
    # Create a figure with a Pie Chart and a Bar Chart side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Pie Chart - Top 10 by Number of Sales
    top_10 = df[colname].value_counts().head(10)
    axes[0].pie(
        top_10,
        labels=[f'{label_name_dict[colname]} {s}' for s in top_10.index],
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.Set3.colors[:len(top_10)]
    )
    axes[0].set_title(f'Top 10 {label_name_dict[colname]} by Number of Sales (Pie Chart)', fontsize=14)

    # 2. Bar Chart - Top 10 by Count of Items Sold
    top_10_by_sales = df[[colname,"item_cnt_day"]].groupby(colname).sum().sort_values("item_cnt_day",ascending=False).head(10).reset_index()
    axes[1].bar(            
        [f'{label_name_dict[colname]} {c}' for c in top_10_by_sales[colname]],
        top_10_by_sales["item_cnt_day"],
        color='cornflowerblue',
        edgecolor='navy'
    )
    axes[1].set_title(f'Top 10 {label_name_dict[colname]} by Number of Items Sold (Bar Chart)', fontsize=14)
    axes[1].set_xlabel(f'{label_name_dict[colname]} ID', fontsize=12)
    axes[1].set_ylabel('Number of Items Sold', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
