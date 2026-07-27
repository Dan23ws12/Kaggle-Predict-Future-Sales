
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from dotenv import load_dotenv
import os
import seaborn as sns
load_dotenv()


cat_cols = ["shop_id", "item_id", "item_category_id", "month_block_num"]
numeric_cols = ["item_price", "item_cnt_month"]
target_col = "item_cnt_month"
sales = pd.read_csv(os.getenv("CLEAN_DATA_PATH") + "/sales_train_cnt_0.csv")

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
    corr_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols)
    num_cat_vars = len(cat_cols)
    corr_values = {}
    for k in range(num_cat_vars):
        corr_values[k] = []

    for i in range(num_cat_vars):
        for j in range(i, num_cat_vars):
            vari, varj = [cat_cols[i], cat_cols[j]]
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
        corr_matrix[cat_cols[k]] = corr_values.get(k)
    return corr_matrix

def get_charts_for_cat_vars(df:pd.DataFrame, colname:str):
    """
    Plots a pie chart and bar chart for a given categorical variable
    """
    if colname not in cat_cols:
        raise ValueError("Column is not a categorical variable")
    label_name_dict = {
        "shop_id": "Shop",
        "item_id": "Item",
        "item_category_id": "Item Category",
        "month_block_num": "Month"
    }
    # Create a figure with a Pie Chart and a Bar Chart side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Pie Chart - Top 5 Shops Share
    top_shops = df[colname].value_counts().head(10)
    axes[0].pie(
        top_shops,
        labels=[f'{label_name_dict[colname]} {s}' for s in top_shops.index],
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.Set3.colors[:len(top_shops)]
    )
    axes[0].set_title(f'Top 10 {colname} by Number of Sales (Pie Chart)', fontsize=14)

    # 2. Bar Chart - Top 10 Item Categories Count
    top_cats = df[[colname,"item_cnt_month"]].groupby(colname).sum().sort_values("item_cnt_month",ascending=False).head(10).reset_index()
    axes[1].bar(
        [f'{label_name_dict[colname]} {c}' for c in top_cats[colname]],
        top_cats["item_cnt_month"],
        color='cornflowerblue',
        edgecolor='navy'
    )
    axes[1].set_title(f'Top 10 {colname} by Number of Items Sold (Bar Chart)', fontsize=14)
    axes[1].set_xlabel(f'{label_name_dict[colname]} ID', fontsize=12)
    axes[1].set_ylabel('Number of Items Sold', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
